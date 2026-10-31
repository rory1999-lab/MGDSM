"""Architecture settings for the public modeling components."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    input_dim: int = 52
    hidden_dim: int = 32
    embed_dim: int = 25
    graph_top_k: int = 5
    logvar_min: float = -6.0
    logvar_max: float = 3.0

    def __post_init__(self) -> None:
        for name in ("input_dim", "hidden_dim", "embed_dim"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.graph_top_k, int) or isinstance(self.graph_top_k, bool) or self.graph_top_k < 0:
            raise ValueError("graph_top_k must be a nonnegative integer")
        if not self.logvar_min < self.logvar_max:
            raise ValueError("logvar_min must be smaller than logvar_max")
