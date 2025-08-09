from .unet.unet_seq import UNetSequential
from .positional.posnet import PosNet

MODEL_REGISTRY = {
    "unet_seq": UNetSequential,
    "posnet": PosNet
}
