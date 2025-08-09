import torch
import torch.nn as nn
import torch.nn.functional as F


def build_normalized_grid(H, W):
    x = torch.linspace(-1, 1, W)
    y = torch.linspace(-1, 1, H)
    grid = torch.stack(torch.meshgrid(x, y, indexing='xy'), dim=-1)
    return grid  # shape: (H, W, 2)


class FFBlock(nn.Module):

    def __init__(self, in_dim, hidden_dim=1024):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 2),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.ffn(x)


class ConvBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, groups=32),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 8, kernel_size=3, padding=1, groups=8),
            nn.LeakyReLU(),
        )
    
    def forward(self, x):
        return self.conv(x)


class PosNet(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv = ConvBlock()
        self.ffn = FFBlock(2+8)

        self.register_buffer(
            "coords", build_normalized_grid(128, 128),
        )

    def forward(self, I_swirl):
        B, C, H, W = I_swirl.shape

        # The conv block extracts F features per pixel
        feats = self.conv(I_swirl).permute(0, 2, 3, 1) # (B,H,W,F)

        # Initialize input coordinates of pixels
        coords = self.coords.unsqueeze(0).repeat(B,1,1,1) #(B,H,W,2)

        # For each pixel, concatenate the coordinates with the F features
        in_coords = torch.concat([coords, feats], dim=-1) #(B, H, W, 2+F)

        # Get the new coordinates as the original coordinates plus 
        # the residual coordinates learnt by the feed-forward block
        out_coords = coords + self.ffn(in_coords) #(B,H,W,2)

        # Use the new coordinates to reconstruct the non-swirled image
        I_recon = F.grid_sample(I_swirl, out_coords, align_corners=True)
        
        return I_recon
    
    def loss_function(self, x, y):
        loss = F.mse_loss(x, y)
        return loss, {'Loss/Full': loss.item()}

    def patchwise_loss(self, x, y, patch_size=16, top_k=16):
        """
        Computes the loss function as the average MSE over the `top_k` patches with the highest MSE.

        We first divide the output of the model into patches and then compute the MSE on each patch.
        Then the loss is average over the `top_k` patches with the highest MSE.
        
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

        return top_k_loss,  {'Loss/Full': F.mse_loss(output, y).item(), 'Loss/TopK': top_k_loss.item(), 'TopK': K}
