"""PatchCore model training wrapper."""
import sys
import types
import warnings

warnings.filterwarnings("ignore")
# Stub out OpenVINO so Anomalib loads even if OpenVINO isn't installed
sys.modules["openvino"] = types.ModuleType("openvino")
dummy = types.ModuleType("dummy")
dummy.OVDict = dict
sys.modules["openvino.runtime.utils.data_helpers.wrappers"] = dummy

from anomalib.models.image.patchcore import Patchcore  # noqa: E402
from lightning.pytorch import Trainer  # noqa: E402


def train_patchcore(data_module, backbone: str = "wide_resnet50_2", epochs: int = 1):
    """Train (or fit) a PatchCore model on the provided LightningDataModule."""
    model = Patchcore(backbone=backbone, pre_trained=True)
    Trainer(
        max_epochs=epochs,
        accelerator="auto",
        num_sanity_val_steps=0,
    ).fit(model, data_module)
    return model
