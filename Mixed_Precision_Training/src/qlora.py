from pathlib import Path
import torch
import math

from .bignet import BIGNET_DIM, LayerNorm
from .low_precision import Linear4Bit  # Ensure this is the 4-bit quantized linear layer

# Set the correct device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


class LoRALinear(Linear4Bit):
    def __init__(self, in_features: int, out_features: int, lora_dim: int, bias: bool = True, lora_alpha: float = 0.0937):
        super(Linear4Bit, self).__init__()
        super().__init__(in_features, out_features, bias=bias)  

        # Ensure lora_a and lora_b are on the correct device
        self.lora_a = torch.nn.Linear(in_features, lora_dim, bias=False, dtype=torch.float32).to(device)
        self.lora_b = torch.nn.Linear(lora_dim, out_features, bias=False, dtype=torch.float32).to(device)

        torch.nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_b.weight)

        self.lora_alpha = lora_alpha  # Scaling factor

        # Ensure correct gradient behavior
        self.requires_grad_(False)  # Freeze base model weights
        self.lora_a.requires_grad_(True)
        self.lora_b.requires_grad_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device)  # Ensure input is on the correct device
        base_output = super().forward(x)  # Uses dequantized weight internally
        lora_output = self.lora_b(self.lora_a(x.to(dtype=torch.float32)))
        return (base_output + self.lora_alpha * lora_output).to(dtype=torch.float32)


class QLoRABigNet(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, channels: int, lora_dim: int, lora_alpha: float):
            super().__init__()
            self.model = torch.nn.Sequential(
                LoRALinear(channels, channels, lora_dim, lora_alpha=lora_alpha),
                torch.nn.ReLU(),
                LoRALinear(channels, channels, lora_dim, lora_alpha=lora_alpha),
                torch.nn.ReLU(),
                LoRALinear(channels, channels, lora_dim, lora_alpha=lora_alpha),
            )

        def forward(self, x: torch.Tensor):
            x = x.to(device)  # Ensure input is on the correct device
            return self.model(x) + x

    def __init__(self, lora_dim: int = 32, lora_alpha: float = 0.0937):
        super().__init__()
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM, lora_dim, lora_alpha),
            LayerNorm(BIGNET_DIM).to(device),
            self.Block(BIGNET_DIM, lora_dim, lora_alpha),
            LayerNorm(BIGNET_DIM).to(device),
            self.Block(BIGNET_DIM, lora_dim, lora_alpha),
            LayerNorm(BIGNET_DIM).to(device),
            self.Block(BIGNET_DIM, lora_dim, lora_alpha),
            LayerNorm(BIGNET_DIM).to(device),
            self.Block(BIGNET_DIM, lora_dim, lora_alpha),
            LayerNorm(BIGNET_DIM).to(device),
            self.Block(BIGNET_DIM, lora_dim, lora_alpha),
        ).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device)  # Ensure input is on the correct device
        return self.model(x)


def load(path: Path | None) -> QLoRABigNet:
    net = QLoRABigNet().to(device)
    if path is not None:
        net.load_state_dict(torch.load(path, map_location=device), strict=False)
    return net




