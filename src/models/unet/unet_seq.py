import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 
            kernel_size=3, padding=1
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        return out

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        self.enc1 = ConvBlock(in_channels, 32) 
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)

        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(128, 64)
        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(64, 32)
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

        self.pool = nn.MaxPool2d(2)
    

    def forward(self, x):

        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        dec3 = self.dec3(torch.cat([self.upconv3(enc3), enc2], dim=1))
        dec2 = self.dec2(torch.cat([self.upconv2(dec3), enc1], dim=1))

        return x + self.final_conv(dec2)

class UNetSequential(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_blocks=5):
        super().__init__()
        
        self.main = nn.Sequential(
            *[UNet(in_channels, out_channels) for _ in range(num_blocks)]
        )

    def forward(self, x):
        return self.main(x)

    def loss_function(self, x, y):
        """
        Computes the loss function for the UNet model.

        Args:
            x (torch.Tensor): Input tensor.
            y (torch.Tensor): Target tensor.
        
        Returns:
            torch.Tensor: Computed loss.
        """
        loss = F.mse_loss(x,y)
        return loss, {"Loss": loss.item()}


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
    

    def loss_function_patch_edge(self, x, y, patch_size=16, top_k=16):
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

        # reshape to (B, N, C*P*P)
        N = out_patches.shape[-1] # number of patches
        out_patches = out_patches.transpose(1, 2)
        y_patches = y_patches.transpose(1, 2)

        # compute MSE per patch
        mse_per_patch = F.mse_loss(out_patches, y_patches, reduction='none')  # (B, N, C*P*P)
        mse_per_patch = mse_per_patch.mean(dim=-1)  # (B, N)

        # Get the indices of the K patches with the highest MSE
        top_k_indices = mse_per_patch.topk(K, dim=1).indices

        

        # compute patch-wise edge loss
        edge_x = sobel_filter_rgb(output)
        edge_y = sobel_filter_rgb(y)
        edge_out_patches = unfold(edge_x).transpose(1, 2)
        edge_y_patches = unfold(edge_y).transpose(1, 2)
        edge_mse_per_patch = F.mse_loss(edge_out_patches, edge_y_patches, reduction='none')  # (B, N, C*P*P)
        edge_mse_per_patch = edge_mse_per_patch.mean(dim=-1)  # (B, N)
        edge_top_k_indices = edge_mse_per_patch.topk(K, dim=1).indices
        edge_top_k_mse = edge_mse_per_patch.gather(1, edge_top_k_indices)
        edge_top_k_loss = edge_top_k_mse.mean()

        # Average the MSE over the top K patches
        top_k_mse = mse_per_patch.gather(1, edge_top_k_indices)
        top_k_loss = top_k_mse.mean()

        loss =  edge_top_k_loss + top_k_loss

        return loss, {
            'Loss': F.mse_loss(output, y).item(),
            'Loss/TopK': top_k_loss.item(),
            'Loss/TopKEdge': edge_top_k_loss.item(),
            'Loss/Combo': loss.item(),
            'TopK': top_k
        }


def sobel_filter_rgb(image: torch.Tensor) -> torch.Tensor:
    """
    Apply Sobel edge detection to a batch of RGB images.

    Args:
        image (torch.Tensor): Tensor of shape (B, 3, H, W)

    Returns:
        torch.Tensor: Edge magnitude map of shape (B, 1, H, W)
    """
    B, C, H, W = image.shape
    assert C == 3, "Input must have 3 channels (RGB)"

    # Define Sobel kernels (3x3) for each channel
    sobel_x = torch.tensor([[-1., 0., 1.],
                            [-2., 0., 2.],
                            [-1., 0., 1.]], device=image.device)

    sobel_y = torch.tensor([[-1., -2., -1.],
                            [ 0.,  0.,  0.],
                            [ 1.,  2.,  1.]], device=image.device)

    # Expand to shape (3, 1, 3, 3) for depthwise conv
    sobel_x = sobel_x.expand(3, 1, 3, 3)
    sobel_y = sobel_y.expand(3, 1, 3, 3)

    # Apply depthwise convolution
    grad_x = F.conv2d(image, sobel_x, padding=1, groups=3)
    grad_y = F.conv2d(image, sobel_y, padding=1, groups=3)

    # Compute gradient magnitude per channel
    magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)

    # Average over RGB channels
    magnitude = magnitude.mean(dim=1, keepdim=True)

    return magnitude