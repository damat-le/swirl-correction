import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.unet.unet_seq import UNetSequential


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
        self.act_fn = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act_fn(out)
        out = self.conv2(out)
        out = self.act_fn(out)
        return out

class SwirlDetectorNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        self.enc1 = ConvBlock(in_channels, 32) 
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.enc4 = ConvBlock(128, 256)

        self.upconv4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(256, 128)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(128, 64)
        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(64, 32)
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

        self.pool = nn.MaxPool2d(2)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):

        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        dec4 = self.dec4(torch.cat([self.upconv4(enc4), enc3], dim=1))
        dec3 = self.dec3(torch.cat([self.upconv3(dec4), enc2], dim=1))
        dec2 = self.dec2(torch.cat([self.upconv2(dec3), enc1], dim=1))

        return self.sigmoid(self.final_conv(dec2))

    def loss_function(self, outputs, mask):
        loss =  F.binary_cross_entropy(outputs, mask)    
        return loss, {'Loss': loss.item()}


class SwirlCorrectorNet(UNetSequential):
    def __init__(self, in_channels=3, out_channels=3, num_blocks=5):
        super().__init__(in_channels, out_channels, num_blocks)

    def forward(self, x):
        return super().forward(x)

    def loss_function(self, outputs, targets, mask):
        outputs = outputs * mask
        targets = targets * mask
        loss =  F.mse_loss(outputs, targets)    
        return loss, {'Loss': loss.item()}


class Pipeline(nn.Module):
    def __init__(self, kwargs_detector={}, kwargs_corrector={}):
        super().__init__()

        self.detector = SwirlDetectorNet(**kwargs_detector)
        self.corrector = SwirlCorrectorNet(**kwargs_corrector)

    def forward(self, swirled_im):
        pred_mask = self.detector(swirled_im)
        corrected = self.corrector(swirled_im)
        return pred_mask, corrected

    def loss_function(self, outputs, targets, mask):
        pred_mask, corrected = outputs
        loss_det, _ = self.detector.loss_function(pred_mask, mask)
        loss_corr, _ = self.corrector.loss_function(corrected, targets, pred_mask)
        loss = loss_det + loss_corr
        return loss, {'Loss/Detector': loss_det.item(), 'Loss/Corrector': loss_corr.item(), 'Loss': loss.item()}