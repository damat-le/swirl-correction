from .unet.unet_seq import UNetSequential
from .positional.posnet import PosNet
from .pipeline.e2e import Pipeline

MODEL_REGISTRY = {
    "unet_seq": UNetSequential,
    "posnet": PosNet,
    "pipeline": Pipeline
}
