import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """
    Variant of Layer Normalization.

    This layer norm accepts to data formats: "channel_first" and "channel_last".

    In "channel_first" mode, the input tensor is expected to have shape (B, C, H, W), and layer normalization is applied across the channel dimension (C) for each spatial location (H, W).

    In "channel_last" mode, the input tensor is expected to have shape (B, H, W, C), and layer normalization is applied across the channel dimension (C) for each spatial location (H, W).
    """
    
    def __init__(self, normalized_shape, data_format="channels_last", **kwargs):
        super().__init__()
        self.data_format = data_format
        self.layer_norm = nn.LayerNorm(normalized_shape, **kwargs)

    def forward(self, x):
        if self.data_format == "channels_first":
            x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
            x = self.layer_norm(x)
            x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        else:
            x = self.layer_norm(x)
        return x


class ConvNextBlock(nn.Module):
    """
    ConvNext Block.

    It receives a tensor of shape (B, C, H, W) as input and outputs a tensor of the same shape.

    The idea is to first apply a depthwise separable convolution, followed by a pointwise convolution.
    The depthwise convolution captures spatial correlations within each channel separately.
    The pointwise convolution mixes information across channels, capturing cross-channel correlations.
    The pointwise convolution is implemented as a Linear layer.
    
    Parameters
    ----------
    dim : int
        The number of input channels.
        The ConvNextBlock output will have the same number of channels.

    res_scale_init_value : float, default=1e-6 
        The initial value for the residual scaling factor.

    """

    def __init__(self, dim, res_scale_init_value=1e-6):
        super().__init__()
        self.dim = dim
        self.res_scale_init_value = res_scale_init_value

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        if res_scale_init_value > 0:
            # Initialize the residual scaling factor
            self.gamma = nn.Parameter(
                res_scale_init_value * torch.ones((dim)), 
                requires_grad=True
            )
        else:
            self.gamma = None


    def forward(self, x):
        """
        Forward pass of the ConvNextBlock.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, H, W).
        """
        res = x

        # Depthwise convolution
        x = self.dwconv(x)

        # Layer normalization and pointwise convolution
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.norm(x)
        x = self.pwconv(x)

        # Eventually apply residual scaling
        if self.gamma is not None:
            x = self.gamma * x
        
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        
        # Residual connection
        x = x + res  

        return x


class UNext(nn.Module):


    def __init__(
            self, 
            in_dim, out_dim, 
            depths, 
            dims, 
            res_scale_init_value=1e-6
        ):
        super().__init__()

        self.stage1 = nn.Conv2d(in_dim, in_dim, kernel_size=7, padding=3, groups=in_dim) 

        self.down1 = nn.Sequential(
            LayerNorm(in_dim, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(in_dim, dims[0], kernel_size=2, stride=2)
        )
        self.stage2 = nn.Sequential(
            *[ConvNextBlock(dims[0], res_scale_init_value=res_scale_init_value) for _ in range(depths[1])]
        )
        self.down2 = nn.Sequential(
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
            nn.Conv2d(dims[0], dims[1], kernel_size=2, stride=2)
        )
        self.stage3 = nn.Sequential(
            *[ConvNextBlock(dims[1], res_scale_init_value=res_scale_init_value) for _ in range(depths[2])]
        )
        self.up2 = nn.Sequential(
            LayerNorm(dims[1], eps=1e-6, data_format="channels_first"),
            nn.ConvTranspose2d(dims[1], dims[0], kernel_size=2, stride=2),
        )
        self.merge2 = nn.Conv2d(dims[0]*2, dims[0], kernel_size=1)
        self.stage6 = nn.Sequential(
            *[ConvNextBlock(dims[0], res_scale_init_value=res_scale_init_value) for _ in range(depths[1])]
        )
        self.up3 = nn.Sequential(
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
            nn.ConvTranspose2d(dims[0], out_dim, kernel_size=2, stride=2),
        )
        self.merge3 = nn.Conv2d(out_dim*2, out_dim, kernel_size=1)
        self.stage7 = nn.Conv2d(out_dim, out_dim, kernel_size=7, padding=3, groups=out_dim)

    def forward(self, x):

        res = x.clone()

        # Encoder
        enc1 = self.stage1(x)
        enc2 = self.stage2(self.down1(enc1))
        enc3 = self.stage3(self.down2(enc2))

        dec2 = self.up2(enc3)
        dec2 = self.merge2(torch.cat([dec2, enc2], dim=1))
        dec2 = self.stage6(dec2)

        dec1 = self.up3(dec2)
        dec1 = self.merge3(torch.cat([dec1, enc1], dim=1))
        dec1 = self.stage7(dec1)

        return res + dec1

    def loss_function(self, x, y):
        """
        Computes the loss function for the UNet model.

        Args:
            x (torch.Tensor): Input tensor.
            y (torch.Tensor): Target tensor.
        
        Returns:
            torch.Tensor: Computed loss.
        """
        loss = F.mse_loss(x, y)
        return loss, {'Loss': loss.item()}

    def loss_function_patch(self, x, y, patch_size=16, top_k=16):
        """
        Computes the loss function for the ResNet model.

        We first divide the output of the model into patches and then compute the MSE on each patch.
        Then the loss is average over the K patches with the highest MSE.
        
        Args:
            x (torch.Tensor): Input tensor.
            y (torch.Tensor): Target tensor.
        
        Returns:
            torch.Tensor: Computed loss.
        """
        P = patch_size
        K = top_k
        B, C, H, W = x.shape

        output = x

        # shape: (B, C * patch_size * patch_size, N)
        unfold = nn.Unfold(
            kernel_size=(P, P), 
            stride=P
        )

        # Get patches from the output and target tensors
        out_patches = unfold(output)
        y_patches = unfold(y)

        # reshape to (B, N, C, P, P)
        N = out_patches.shape[-1] # number of patches
        out_patches = out_patches.transpose(1, 2)#.reshape(B, N, C, P, P)
        y_patches = y_patches.transpose(1, 2)#.reshape(B, N, C, P, P)

        # compute MSE per patch
        mse_per_patch = F.mse_loss(out_patches, y_patches, reduction='none')  # (B, N, C, P, P)
        # mse_per_patch = mse_per_patch.view(B, N, -1).mean(dim=-1)  # (B, N)
        mse_per_patch = mse_per_patch.mean(dim=-1)  # (B, N)

        # Get the indices of the K patches with the highest MSE
        top_k_indices = mse_per_patch.topk(K, dim=1).indices

        # Average the MSE over the top K patches
        top_k_mse = mse_per_patch.gather(1, top_k_indices)
        top_k_loss = top_k_mse.mean()

        return top_k_loss, {
            'Loss': F.mse_loss(output, y).item(),
            'Loss/TopK': top_k_loss.item(),
            'TopK': top_k
        }
    

    