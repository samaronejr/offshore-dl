"""Model implementations — all inherit from BaseModel."""

from offshore_dl.models.base import BaseModel, model_summary
from offshore_dl.models.chronos_wrapper import ChronosWrapper
from offshore_dl.models.convtran import ConvTranModel
from offshore_dl.models.deeponet import DeepONetModel
from offshore_dl.models.dummy import DummyModel
from offshore_dl.models.inception_time import InceptionTimeModel
from offshore_dl.models.lstm import LSTMModel
from offshore_dl.models.tcn import TCNModel

__all__ = [
    "BaseModel",
    "ChronosWrapper",
    "ConvTranModel",
    "DeepONetModel",
    "DummyModel",
    "InceptionTimeModel",
    "LSTMModel",
    "TCNModel",
    "model_summary",
]

# Optional imports — these require extra dependencies
try:
    from offshore_dl.models.patchtst import PatchTSTModel as PatchTSTModel

    __all__.append("PatchTSTModel")
except (ImportError, ModuleNotFoundError, RuntimeError):
    pass

# Optional imports — these require extra dependencies
try:
    from offshore_dl.models.timesfm_wrapper import TimesFMWrapper as TimesFMWrapper

    __all__.append("TimesFMWrapper")
except ImportError:
    pass

try:
    from offshore_dl.models.tirex_wrapper import TiRexWrapper as TiRexWrapper

    __all__.append("TiRexWrapper")
except ImportError:
    pass

try:
    from offshore_dl.models.fkmad import FKMADModel as FKMADModel

    __all__.append("FKMADModel")
except (ImportError, ModuleNotFoundError, RuntimeError):
    pass

try:
    from .mambasl import MambaSLModel as MambaSLModel

    __all__.append("MambaSLModel")
except (ImportError, ModuleNotFoundError, RuntimeError):
    pass

try:
    from .convtimenet import ConvTimeNetModel as ConvTimeNetModel

    __all__.append("ConvTimeNetModel")
except (ImportError, ModuleNotFoundError, RuntimeError):
    pass

try:
    from offshore_dl.models.hydra_rocket import HydraRocketModel as HydraRocketModel

    __all__.append("HydraRocketModel")
except ImportError:
    pass
