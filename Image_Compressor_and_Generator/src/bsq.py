import abc

import torch

from .ae import PatchAutoEncoder


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "BSQPatchAutoEncoder"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


def diff_sign(x: torch.Tensor) -> torch.Tensor:
    """
    A differentiable sign function using the straight-through estimator.
    Returns -1 for negative values and 1 for non-negative values.
    """
    sign = 2 * (x >= 0).float() - 1
    return x + (sign - x).detach()


class Tokenizer(abc.ABC):
    """
    Base class for all tokenizers.
    Implement a specific tokenizer below.
    """

    @abc.abstractmethod
    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tokenize an image tensor of shape (B, H, W, C) into
        an integer tensor of shape (B, h, w) where h * patch_size = H and w * patch_size = W
        """

    @abc.abstractmethod
    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode a tokenized image into an image tensor.
        """


class BSQ(torch.nn.Module):
    def __init__(self, codebook_bits: int, embedding_dim: int):
        super().__init__()
        self._codebook_bits = codebook_bits
        self._embedding_dim = embedding_dim
        
        # Linear projections for encoding and decoding
        self.down_proj = torch.nn.Linear(64, codebook_bits)
        self.up_proj = torch.nn.Linear(codebook_bits, 64)


    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implement the BSQ encoder:
        - A linear down-projection into codebook_bits dimensions
        - L2 normalization
        - differentiable sign
        """
        # Down-project to codebook_bits dimensions
        x = self.down_proj(x)
        
        # L2 normalization
        x = torch.nn.functional.normalize(x, p=2, dim=-1)
        
        # Differentiable sign for binary encoding
        x = diff_sign(x)
        
        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implement the BSQ decoder:
        - A linear up-projection into embedding_dim should suffice
        """
        # Up-project back to embedding_dim
        x = self.up_proj(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run BQS and encode the input tensor x into a set of integer tokens
        """
        return self._code_to_index(self.encode(x))

    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode a set of integer tokens into an image.
        """
        return self.decode(self._index_to_code(x))

    def _code_to_index(self, x: torch.Tensor) -> torch.Tensor:
        x = (x >= 0).int()
        return (x * 2 ** torch.arange(x.size(-1)).to(x.device)).sum(dim=-1)

    def _index_to_code(self, x: torch.Tensor) -> torch.Tensor:
        return 2 * ((x[..., None] & (2 ** torch.arange(self._codebook_bits).to(x.device))) > 0).float() - 1


class BSQPatchAutoEncoder(PatchAutoEncoder, Tokenizer):
    """
    Combine your PatchAutoEncoder with BSQ to form a Tokenizer.

    Hint: The hyper-parameters below should work fine, no need to change them
          Changing the patch-size of codebook-size will complicate later parts of the assignment.
    """

    def __init__(self, patch_size: int = 5, latent_dim: int = 128, codebook_bits: int = 10):
        super().__init__(patch_size=patch_size, latent_dim=latent_dim)
        self.codebook_bits = codebook_bits
        self.bsq = BSQ(codebook_bits=codebook_bits, embedding_dim=latent_dim)

    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        # First use the PatchAutoEncoder's encoder, then the BSQ encoding to indices
        features = super().encode(x)
        return self.bsq.encode_index(features)

    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        # First decode indices to features, then use the decoder
        features = self.bsq.decode_index(x)
        return super().decode(features)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # Encode using PatchAutoEncoder and then apply BSQ encoding
        features = super().encode(x)
        return self.bsq.encode(features)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        # Decode using BSQ and then apply PatchAutoEncoder decoding
        features = self.bsq.decode(x)
        return super().decode(features)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Return the reconstructed image and a dictionary of additional loss terms you would like to
        minimize (or even just visualize).
        Hint: It can be helpful to monitor the codebook usage with

              cnt = torch.bincount(self.encode_index(x).flatten(), minlength=2**self.codebook_bits)

              and returning

              {
                "cb0": (cnt == 0).float().mean().detach(),
                "cb2": (cnt <= 2).float().mean().detach(),
                ...
              }
        """
        # Encode the input
        features = super().encode(x)
        
        # Apply BSQ encoding and decoding
        binary_encoding = self.bsq.encode(features)
        decoded_features = self.bsq.decode(binary_encoding)
        
        # Decode back to image space
        reconstruction = super().decode(decoded_features)
        
        # Monitor codebook usage
        indices = self.bsq.encode_index(features)
        cnt = torch.bincount(indices.flatten(), minlength=2**self.codebook_bits)
        
        # Collect metrics
        metrics = {
            "cb0": (cnt == 0).float().mean().detach(),  # Unused codebook entries
            "cb2": (cnt <= 2).float().mean().detach(),  # Rarely used codebook entries
            "cb10": (cnt <= 10).float().mean().detach(),  # Infrequently used entries
            "max_usage": cnt.max().detach(),  # Most frequent entry count
        }
        
        return reconstruction, metrics

