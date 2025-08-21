from .unet.unet_seq import UNetSequential
from .positional.posnet import PosNet
from .pipeline.e2e import Pipeline
from .convnext.convnext import ConvNextFull, ConvNextSeq
from .unext.unext import UNext

MODEL_REGISTRY = {
    "unet_seq": UNetSequential,
    "posnet": PosNet,
    "pipeline": Pipeline,
    "convnext": ConvNextFull,
    "convnext_seq": ConvNextSeq,
    "unext": UNext
}
