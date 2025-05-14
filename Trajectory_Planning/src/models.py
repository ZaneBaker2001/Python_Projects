from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]




class MLPPlanner(nn.Module):
    def __init__(self, n_track=10, n_waypoints=3):
        super(MLPPlanner, self).__init__()
        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        input_dim = n_track * 4  # 10 points each for left and right boundaries, each point has 2 coordinates
        hidden_dim = 64  # Hidden layer size, adjust to balance performance and model size
        output_dim = n_waypoints * 2  # Each waypoint has 2 coordinates (x, y)

        # Define a small MLP with two hidden layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, track_left, track_right):
        # Flatten the track boundary inputs and concatenate
        x = torch.cat((track_left, track_right), dim=-1)  # Shape: (B, n_track, 4)
        x = x.view(x.size(0), -1)  # Flatten to shape (B, input_dim)

        # Forward pass through the MLP
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # Reshape output to (B, n_waypoints, 2)
        return x.view(x.size(0), -1, 2)

# Define a suitable loss function (e.g., Mean Squared Error)
def waypoint_loss(predicted, target, mask):
    # Apply mask to only include clean waypoints
    masked_predicted = predicted * mask.unsqueeze(-1)
    masked_target = target * mask.unsqueeze(-1)
    loss = F.mse_loss(masked_predicted, masked_target)
    return loss


















class TransformerPlanner(nn.Module):
    def __init__(self, n_waypoints=3, d_model=128, nhead=2, num_decoder_layers=2, num_encoder_layers=2, dropout=0.1):
        super(TransformerPlanner, self).__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        self.n_waypoints = n_waypoints
        self.d_model = d_model

        # Linear layer to project input from dimension 4 to d_model (128) with dropout
        self.input_proj = nn.Sequential(
            nn.Linear(4, d_model),
            nn.Dropout(dropout)
        )

        # Embedding layer for waypoints (query embeddings)
        self.waypoint_embed = nn.Embedding(n_waypoints, d_model)

        # Transformer encoder for lane boundaries with dropout
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)

        # Transformer decoder for waypoints with dropout
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256, dropout=dropout
        )
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_decoder_layers)

        # Final fully connected layer to map to (n_waypoints * 2) with dropout
        self.fc = nn.Sequential(
            nn.Linear(d_model, 2),
            nn.Dropout(dropout)
        )

    def forward(self, track_left, track_right):
        """
        Forward pass for the TransformerPlanner.

        Args:
        - track_left (Tensor): Left lane boundary points, shape (B, n_track, 2)
        - track_right (Tensor): Right lane boundary points, shape (B, n_track, 2)

        Returns:
        - waypoints (Tensor): Predicted waypoints of shape (B, n_waypoints, 2)
        """
        # Concatenate left and right track boundaries (B, n_track, 4)
        track_input = torch.cat([track_left, track_right], dim=-1)

        # Project input to match transformer embedding dimension
        track_input = self.input_proj(track_input)  # Shape (B, n_track, d_model)

        # (B, n_track, d_model) -> (n_track, B, d_model) to match Transformer input shape
        track_input = track_input.permute(1, 0, 2)

        # Pass through Transformer Encoder
        track_encoded = self.transformer_encoder(track_input)

        # Create waypoint query embeddings (n_waypoints, B, d_model)
        waypoint_queries = self.waypoint_embed.weight.unsqueeze(1).expand(-1, track_input.size(1), -1)

        # Pass through Transformer Decoder
        decoder_output = self.transformer_decoder(waypoint_queries, track_encoded)

        # Pass through the final fully connected layer
        output = self.fc(decoder_output)

        # Reshape to (B, n_waypoints, 2)
        output = output.permute(1, 0, 2)

        return output






class CNNPlanner(nn.Module):
    def __init__(self, n_waypoints: int = 3, input_height: int = 96, input_width: int = 128):
        super(CNNPlanner, self).__init__()
        
        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))
        
        self.n_waypoints = n_waypoints
        
        # Efficient convolutional layers with fewer filters and separable convolutions
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1, groups=8)  # Depthwise
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, groups=16)  # Depthwise
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((input_height // 16, input_width // 16))

        feature_height = input_height // 16
        feature_width = input_width // 16
        flattened_dim = 64 * feature_height * feature_width

        # Smaller fully connected layers
        self.fc1 = nn.Linear(flattened_dim, 64)
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(64, n_waypoints * 2)

        self.relu = nn.ReLU()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(image)))
        x = self.pool(x)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.avg_pool(x)

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x.view(x.size(0), self.n_waypoints, 2)























MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
