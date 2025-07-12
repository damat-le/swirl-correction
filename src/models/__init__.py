from .ae.ae import Autoencoder
from .unet.unet import UNet
from .resnet.resnet import ResNet, ResNetPatch
from .random.random import RandomModel

MODEL_REGISTRY = {
    "autoencoder": Autoencoder,
    "unet": UNet,
    "resnet": ResNet,
    "resnet_patch": ResNetPatch,
    "random": RandomModel
}
