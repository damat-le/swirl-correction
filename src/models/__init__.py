from .ae.ae import Autoencoder
from .unet.unet import UNet
from .unet.unet_seq import UNetSequential
from .resnet.resnet import ResNet, ResNetPatch
from .random.random import RandomModel

MODEL_REGISTRY = {
    "autoencoder": Autoencoder,
    "unet": UNet,
    "unet_seq": UNetSequential,
    "resnet": ResNet,
    "resnet_patch": ResNetPatch,
    "random": RandomModel
}
