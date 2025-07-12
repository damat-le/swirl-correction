import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=256):
        super(Autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, 
                out_channels=16,
                kernel_size=3, 
                stride=2, 
                padding=1
                ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=16, 
                out_channels=32,
                kernel_size=3, 
                stride=2, 
                padding=1
                ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=32, 
                out_channels=64,
                kernel_size=3, 
                stride=2, 
                padding=1
                ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=64, 
                out_channels=128,
                kernel_size=3, 
                stride=2, 
                padding=1
                ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=128, 
                out_channels=256,
                kernel_size=3, 
                stride=2, 
                padding=1
                ),
            nn.LeakyReLU(),
        )
        
        dummy_input = torch.randn(1, in_channels, 128, 128)
        self.enc_outdim = self.get_encoder_outdim(dummy_input)
        new_dim = torch.prod(torch.tensor(self.enc_outdim[1:]))

        self.to_latent = nn.Linear(new_dim, hidden_dim)
        self.from_latent = nn.Linear(hidden_dim, new_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                ),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                ),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                ),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=16,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                ),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(
                in_channels=16,
                out_channels=in_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                ),
            nn.Sigmoid()
        )

    def get_encoder_outdim(self, x):
        tmp = self.encoder(x)
        return tmp.shape

    def forward(self, x):
        tmp = self.encoder(x)
        tmp = tmp.view(tmp.size(0), -1)
        z = self.to_latent(tmp)
        tmp = self.from_latent(z)
        tmp = tmp.view(tmp.size(0), *self.enc_outdim[1:])
        rec = self.decoder(tmp)
        return rec, z
