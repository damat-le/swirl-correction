import torch
import torch.nn as nn
from torch.nn import functional as F

class ResNetBlockRevisited(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlockRevisited, self).__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=3, 
            padding=1
        )
        
        self.conv2 = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=3, 
            padding=1
        )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = nn.ReLU()(out)
        out = self.conv2(out)
        out += residual
        out = nn.ReLU()(out)
        return out


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=3, 
            padding=1
        )

        self.batchnorm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 
            kernel_size=3, 
            padding=1
        )

        self.batchnorm2 = nn.BatchNorm2d(out_channels)

        self.convadapter = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1,
            padding=0
        ) if in_channels != out_channels else None

        self.activation = nn.LeakyReLU()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.batchnorm1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.batchnorm2(out)
        residual = self.convadapter(residual) if self.convadapter else residual
        out += residual
        out = self.activation(out)
        return out
    

class ResNet(nn.Module):

    def __init__(self, block_dims):
        super().__init__()
        self.layers = nn.ModuleList()
        for in_channels, out_channels in zip(block_dims[:-1], block_dims[1:]):
            self.layers.append(ResNetBlock(in_channels, out_channels))

        # change activation function to Sigmoid for the final layer
        self.layers[-1].activation = nn.Sigmoid()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def loss_function(self, x, y):
        """
        Computes the loss function for the ResNet model.
        
        Args:
            x (torch.Tensor): Input tensor.
            y (torch.Tensor): Target tensor.
        
        Returns:
            torch.Tensor: Computed loss.
        """
        output = self.forward(x)
        return F.mse_loss(output, y)


class ResNetPatch(nn.Module):

    def __init__(self, block_dims):
        super().__init__()
        self.layers = nn.ModuleList()
        for in_channels, out_channels in zip(block_dims[:-1], block_dims[1:]):
            self.layers.append(ResNetBlock(in_channels, out_channels))

        # change activation function to Sigmoid for the final layer
        self.layers[-1].activation = nn.Sigmoid()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def loss_function(self, x, y, patch_size=16, top_k=16):
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

        return top_k_loss, {'loss_full': F.mse_loss(output, y)}