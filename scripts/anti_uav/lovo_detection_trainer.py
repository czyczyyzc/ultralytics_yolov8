"""Detection trainer used by real-gray LOVO and final mixed training."""

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.torch_utils import strip_optimizer


class LovoDetectionTrainer(DetectionTrainer):
    """Keep per-epoch validation but avoid the redundant final validation pass."""

    def final_eval(self) -> None:
        for checkpoint in (self.last, self.best):
            if checkpoint.exists():
                strip_optimizer(checkpoint)
