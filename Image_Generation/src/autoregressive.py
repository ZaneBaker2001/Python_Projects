import abc

import torch


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "AutoregressiveModel"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


class Autoregressive(abc.ABC):
    """
    Base class for all autoregressive models.
    Implement a specific model below.
    """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Take a tensor x (B, h, w) if integers as input.
        Produce a probability over the next token as an output (B, h, w, n_token).
        Make sure the model is auto-regressive:
          - The first output result[:, 0, 0] does not depend on any input
          - The second output result[:, 0, 1] depends only on x[:, 0, 0]
          - etc.

        Hint 1: Flatten the tensor into a sequence.
        Hint 2: A positional embedding can help, but is not required.
        Hint 3: You need to shift the input sequence by 1 position. Do this after embedding the
                values, and before passing them through your model. (torch.concat or
                torch.nn.ConstantPad1d both work)
        """

    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:  # noqa
        """
        Use your generative model to produce B new token images of size (B, h, w) and type (int/long).
        """


class AutoregressiveModel(torch.nn.Module, Autoregressive):
    """
    Implement an auto-regressive model.
    The input is a set of patch tokens (integers), the output is an image of probability.
    You need to implicitly shift your inputs by one position in the forward pass.
    Make sure n_tokens matches your BSQ dimension (2**codebook_bits_).

    Hint: You will need the torch.nn.Embedding function
    Hint: You can use torch.nn.TransformerEncoderLayer if you'd like
    Hint: You can complete this homework without using positional embeddings
    """

    def __init__(self, d_latent: int = 128, n_tokens: int = 2**10):
        super().__init__()
        self.d_latent = d_latent
        self.n_tokens = n_tokens
        
        # Token embedding layer
        self.token_embedding = torch.nn.Embedding(n_tokens, d_latent)
        
        # Positional embedding (optional but helps with generation quality)
        self.use_pos_embedding = True
        if self.use_pos_embedding:
            self.pos_embedding = torch.nn.Parameter(torch.zeros(1, 1000, d_latent))  # Support sequences up to length 1000
        
        # Transformer layers with causal masking
        nhead = 8  # Number of attention heads
        num_layers = 4  # Number of transformer layers
        
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_latent,
            nhead=nhead,
            dim_feedforward=d_latent * 4,
            dropout=0.1,
            batch_first=True
        )
        
        self.transformer = torch.nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Output projection to token probabilities
        self.output_proj = torch.nn.Linear(d_latent, n_tokens)
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # x shape: (B, h, w)
        batch_size, h, w = x.shape
        
        # Flatten the spatial dimensions to get a sequence
        # New shape: (B, h*w)
        x_flat = x.reshape(batch_size, -1)
        seq_len = x_flat.shape[1]
        
        # Embed the tokens
        # New shape: (B, h*w, d_latent)
        token_embed = self.token_embedding(x_flat)
        
        # Add positional embedding if using
        if self.use_pos_embedding:
            pos_embed = self.pos_embedding[:, :seq_len, :]
            token_embed = token_embed + pos_embed
        
        # Prepare inputs for autoregressive prediction
        # We need to shift the sequence by 1 position for autoregressive modeling
        # For the first position, we'll use a special "start" embedding
        # (can be learned or fixed, here we'll use zeros)
        start_embed = torch.zeros((batch_size, 1, self.d_latent), device=x.device)
        
        # Shift inputs right (remove the last token, add start token at beginning)
        # Shape remains (B, h*w, d_latent)
        shifted_embed = torch.cat([start_embed, token_embed[:, :-1, :]], dim=1)
        
        # Apply transformer with causal mask using the recommended function
        # Generate square subsequent mask as recommended in the assignment
        causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
        transformer_output = self.transformer(shifted_embed, mask=causal_mask)
        
        # Project to token probabilities
        # Shape: (B, h*w, n_tokens)
        logits = self.output_proj(transformer_output)
        
        # Reshape back to (B, h, w, n_tokens)
        logits = logits.reshape(batch_size, h, w, self.n_tokens)
        
        # Return logits and an empty dict for additional info
        return logits, {}

    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:  # noqa
        """
        Generate new token images autoregressively
        """
        if device is None:
            device = next(self.parameters()).device
            
        # Initialize output tensor with zeros (or any other starting token)
        output = torch.zeros((B, h, w), dtype=torch.long, device=device)
        
        # Generate tokens one by one in raster scan order (row by row)
        for i in range(h):
            for j in range(w):
                # Get the current sequence up to position (i, j)
                current_output = output.clone()
                
                # Forward pass to get next token probabilities
                logits, _ = self.forward(current_output)
                
                # Extract probabilities for position (i, j)
                # Shape: (B, n_tokens)
                probs_ij = torch.softmax(logits[:, i, j, :], dim=-1)
                
                # Sample from the distribution
                next_tokens = torch.multinomial(probs_ij, num_samples=1).squeeze(-1)
                
                # Place the sampled tokens at position (i, j)
                output[:, i, j] = next_tokens
        
        return output

