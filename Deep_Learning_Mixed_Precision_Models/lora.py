from pathlib import Path
import torch
import math

BIGNET_DIM = 1024

class LayerNorm(torch.nn.Module):
    """
    torch.nn.LayerNorm is a bit weird with the shape of the input tensor.
    We instead use torch.nn.functional.group_norm with num_groups=1.
    """

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
        r = torch.nn.functional.group_norm(x, 1, self.weight, self.bias, self.eps)
        return r

class LoRALinear(torch.nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_dim: int = 32,
        lora_alpha: float = 1.0,
        bias: bool = True,
    ) -> None:
        super().__init__(in_features, out_features, bias)
        
        # Convert base weights to float16
        self.weight.data = self.weight.data.to(dtype=torch.float16)
        if self.bias is not None:
            self.bias.data = self.bias.data.to(dtype=torch.float16)
        
        # LoRA adapters in float32 for numerical stability
        self.lora_dim = lora_dim
        self.scaling = lora_alpha / lora_dim
        
        self.lora_a = torch.nn.Linear(in_features, lora_dim, bias=False, dtype=torch.float32)
        self.lora_b = torch.nn.Linear(lora_dim, out_features, bias=False, dtype=torch.float32)
        
        # Initialize LoRA layers
        torch.nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_b.weight)
        
        # Freeze base model weights
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base forward in float16
        base_out = torch.nn.functional.linear(
            x.to(dtype=torch.float16), 
            self.weight, 
            self.bias
        )
        
        # LoRA forward in float32
        lora_out = self.lora_b(self.lora_a(x.to(dtype=torch.float32))) * self.scaling
        
        # Combine and return in float32
        return (base_out.to(dtype=torch.float32) + lora_out)

class LoraBigNet(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.model = torch.nn.Sequential(
                LoRALinear(channels, channels),
                torch.nn.ReLU(),
                LoRALinear(channels, channels),
                torch.nn.ReLU(),
                LoRALinear(channels, channels),
            )

        def forward(self, x):
            return self.model(x) + x

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

def load(path: Path | None) -> LoraBigNet:
    net = LoraBigNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True), strict=False)
    return net



