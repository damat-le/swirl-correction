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


class ConvNext(nn.Module):
    """
    ConvNext Architecture.

    The idea is to alternate between downsampling and ConvNext blocks.

    A downsampling block reduces the spatial dimensions (H, W) of the input feature maps and expands the channel dimension (C). When downsampling, this block "patchifies" the input by applying a convolution with a stride equal to the kernel size. This emulates the patch mechanism in Transformers.

    The ConvNext blocks extract features from the downsampled input.
    Each stage consists of several ConvNext blocks applied sequentially.
    The computation across stages is distributed according to the ratio 1:1:3:1, as in small SwinTransformer.

    Parameters
    ----------

    in_dim: int
        The number of input channels.

    out_dim: int
        The number of output channels.

    depths: tuple[int]
        Number of ConvNextBlock at each stage.

    dims: tuple[int]
        Channel dimension at each stage.

    res_scale_init_value: float, default=1e-6
        The initial value for the residual scaling factor.
    """
    
    def __init__(self, in_dim, depths, dims, res_scale_init_value=1e-6):
        super().__init__()
        self.in_dim = in_dim
        self.depths = depths
        self.dims = dims
        self.res_scale_init_value = res_scale_init_value


        assert len(depths) == len(dims), "depths and dims must have the same length"

        # -------------------------------
        # Initialize downsampling blocks
        # -------------------------------

        self.downs = nn.ModuleList()

        stem = nn.Sequential(
            nn.Conv2d(in_dim, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downs.append(stem)

        for i in range(len(dims) - 1):
            down = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downs.append(down)

        # -------------------------------
        # Initialize stages
        # -------------------------------

        self.stages = nn.ModuleList()
        for i in range(len(dims)):
            stage = nn.Sequential(
                *[ConvNextBlock(dims[i], res_scale_init_value=res_scale_init_value) for _ in range(depths[i])]
            )
            self.stages.append(stage)

    def forward(self, x):
        """
        Forward pass of the ConvNet.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, H, W).
        """ 
        for i in range(len(self.dims)):
            x = self.downs[i](x)
            x = self.stages[i](x)
        return x
    

class ConvNextDecoder(nn.Module):
    """
    ConvNext Decoder.

    This decoder is used to upsample the feature maps back to the original input size.
    It consists of a series of ConvNext blocks followed by upsampling operations.

    Parameters
    ----------

    out_dim: int
        The number of output channels.

    depths: tuple[int]
        Number of ConvNextBlock at each stage.

    dims: tuple[int]
        Channel dimension at each stage.
    """
    
    def __init__(self, out_dim, depths, dims, res_scale_init_value=1e-6):
        super().__init__()
        self.out_dim = out_dim
        depths = depths[::-1]
        self.depths = depths
        dims = dims[::-1]
        self.dims = dims

        assert len(depths) == len(dims), "depths and dims must have the same length"

        # -------------------------------
        # Initialize stages
        # -------------------------------

        self.stages = nn.ModuleList()
        for i in range(len(dims)-1):
            stage = nn.Sequential(
                *[ConvNextBlock(dims[i+1], res_scale_init_value=res_scale_init_value) for _ in range(depths[i])]
            )
            self.stages.append(stage)
        
        self.stages.append(
            nn.Sequential(
                *[ConvNextBlock(out_dim, res_scale_init_value=res_scale_init_value) for _ in range(depths[-1])]
            )
        )

        # -------------------------------
        # Initialize upsampling blocks
        # -------------------------------

        self.ups = nn.ModuleList()

        for i in range(len(dims) - 1):
            up = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.ConvTranspose2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.ups.append(up)

        final_up = nn.Sequential(
            LayerNorm(dims[-1], eps=1e-6, data_format="channels_first"),
            nn.ConvTranspose2d(dims[-1], out_dim, kernel_size=4, stride=4),
        )
        self.ups.append(final_up)

    def forward(self, x):
        """
        Forward pass of the ConvNet decoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, H, W).
        """
        for i in range(len(self.dims)):
            x = self.ups[i](x)
            x = self.stages[i](x)
        return x


class ConvNextFull(nn.Module):
    """
    ConvNext Full Architecture.

    This architecture combines the ConvNext encoder and decoder to form a complete model.
    It takes an input tensor, processes it through the encoder, and then upsamples it back to the original size using the decoder.

    Parameters
    ----------
    in_dim: int
        The number of input channels.

    out_dim: int
        The number of output channels.

    depths: tuple[int]
        Number of ConvNextBlock at each stage.

    dims: tuple[int]
        Channel dimension at each stage.
    """

    def __init__(self, in_dim, out_dim, depths, dims, res_scale_init_value=1e-6):
        super().__init__()
        self.encoder = ConvNext(in_dim, depths, dims, res_scale_init_value=res_scale_init_value)
        self.decoder = ConvNextDecoder(out_dim, depths, dims, res_scale_init_value=res_scale_init_value)

    def forward(self, x):
        """
        Forward pass of the ConvNet full architecture.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, H, W).
        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class ConvNextSeq(nn.Module):

    def __init__(self, in_dim, out_dim, res_blocks, depths, dims, res_scale_init_value=1e-6):
        super().__init__()

        self.main = nn.ModuleList()

        for _ in range(res_blocks):
            self.main.append(
                ConvNextFull(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    depths=depths,
                    dims=dims,
                    res_scale_init_value=res_scale_init_value
                )
            )
 
    def forward(self, x):

        res = x
        for layer in self.main:
            x = layer(x)
            x = x + res
            res = x
        return x


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
