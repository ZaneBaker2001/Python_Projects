import abc
import torch
import torch.nn.functional as F


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "PatchAutoEncoder"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


def hwc_to_chw(x: torch.Tensor) -> torch.Tensor:
    dims = list(range(x.dim()))
    dims = dims[:-3] + [dims[-1]] + [dims[-3]] + [dims[-2]]
    return x.permute(*dims)


def chw_to_hwc(x: torch.Tensor) -> torch.Tensor:
    dims = list(range(x.dim()))
    dims = dims[:-3] + [dims[-2]] + [dims[-1]] + [dims[-3]]
    return x.permute(*dims)


class PatchAutoEncoder(torch.nn.Module):
    class PatchEncoder(torch.nn.Module):
        def __init__(self, patch_size: int, latent_dim: int, bottleneck: int):
            super().__init__()
            self.patchify = torch.nn.Conv2d(3, latent_dim, patch_size, patch_size, bias=False)
            self.conv1 = torch.nn.Conv2d(latent_dim, bottleneck, kernel_size=3, padding=1)
            self.act = torch.nn.GELU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = hwc_to_chw(x)
            x = self.patchify(x)
            x = self.act(self.conv1(x))
            return chw_to_hwc(x)

    class PatchDecoder(torch.nn.Module):
        def __init__(self, patch_size: int, latent_dim: int, bottleneck: int):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(bottleneck, latent_dim, kernel_size=3, padding=1)
            self.act = torch.nn.GELU()
            self.unpatchify = torch.nn.ConvTranspose2d(latent_dim, 3, patch_size, patch_size, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = hwc_to_chw(x)
            x = self.act(self.conv1(x))
            x = self.unpatchify(x)
            return chw_to_hwc(x)

    def __init__(self, patch_size: int = 25, latent_dim: int = 128, bottleneck: int = 64):
        super().__init__()
        self.encoder = self.PatchEncoder(patch_size, latent_dim, bottleneck)
        self.decoder = self.PatchDecoder(patch_size, latent_dim, bottleneck)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        loss_terms = {"reconstruction_loss": F.mse_loss(reconstructed, x)}
        return reconstructed, loss_terms

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


