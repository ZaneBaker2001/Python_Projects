from pathlib import Path
from .bignet import BIGNET_DIM
import torch

class LayerNorm(torch.nn.Module):
    num_channels: int
    eps: float

    def __init__(self, num_channels: int, eps: float = 1e-5, affine: bool = True, device=None, dtype=None) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        if affine:
            self.weight = torch.nn.Parameter(torch.empty(num_channels, device=device, dtype=dtype))
            self.bias = torch.nn.Parameter(torch.empty(num_channels, device=device, dtype=dtype))

            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.group_norm(x, 1, self.weight, self.bias, self.eps)

class HalfLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias=bias)
        # Convert parameters to half precision
        self.weight.data = self.weight.data.half()
        if self.bias is not None:
            self.bias.data = self.bias.data.half()
        # Disable gradients for linear layer parameters
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)
    
    def forward(self, x):
        # Forward pass without tracking gradients for linear operations
        with torch.no_grad():
            orig_dtype = x.dtype
            out = super().forward(x.half())
            return out.to(orig_dtype)

class HalfBigNet(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.model = torch.nn.Sequential(
                HalfLinear(channels, channels),
                torch.nn.ReLU(),
                HalfLinear(channels, channels),
                torch.nn.ReLU(),
                HalfLinear(channels, channels),
            )

        def forward(self, x):
            # Forward pass for block operations
            with torch.set_grad_enabled(False):
                out = self.model(x)
            # Only residual connection can have gradients
            return out + x

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
        )

    def forward(self, x):
        return self.model(x)

def load(path: Path | None) -> HalfBigNet:
    net = HalfBigNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True))
    return net



