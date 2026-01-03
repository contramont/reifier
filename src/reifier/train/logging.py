from dataclasses import dataclass, field
from typing import Any
import uuid


@dataclass
class Log:
    """Logs training and validation metrics"""

    data: dict[str, dict[int, float]] = field(default_factory=dict[str, dict[int, float]])  # metric name -> step -> value
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: dict[str, Any] = field(default_factory=dict[str, Any])
    log_dir: str = field(default_factory=lambda: "logs")

    def __post_init__(self):
        self.data['train_loss'] = {}

    def print_step(self, step: int) -> None:
        log_str = f"{step}: "
        for metric, value in self.data.items():
            if step in value:
                log_str += f"{metric}:{value[step]:.4f}  "
        print(log_str)
