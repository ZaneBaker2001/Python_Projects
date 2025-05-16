import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from homework.datasets.road_dataset import load_data
from homework.models import MLPPlanner, save_model
from homework.metrics import PlannerMetric

def train_mlp_planner():
    # Hyperparameters
    train_dataset_path = "/Users/zanebaker/Desktop/Trajectory_Planner/drive_data/train"
    val_dataset_path = "/Users/zanebaker/Desktop/Trajectory_Planner/drive_data/val"
    batch_size = 128  # Actual hardware-limited batch size
      # Larger effective batch size through accumulation
      # Number of accumulation steps
    num_epochs = 50
    learning_rate = 1e-4

    # Target error values
    LON_ERROR_TARGET = 0.16
    LAT_ERROR_TARGET = 0.5

    # Load dataset
    train_loader = load_data(train_dataset_path, batch_size=batch_size, transform_pipeline="aug")
    val_loader = load_data(val_dataset_path, batch_size=batch_size, transform_pipeline="aug")

    # Initialize model
    model = MLPPlanner(n_track=10, n_waypoints=3)
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler to reduce LR on plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Initialize metrics
    train_metric = PlannerMetric()
    val_metric = PlannerMetric()

    best_val_l1_error = float('inf')
    patience = 20
    patience_counter = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_metric.reset()
        total_loss = 0.0
        processed_train_batches = 0  # Counter for batches with valid waypoints

        for batch in train_loader:
            track_left = batch["track_left"].to(device)
            track_right = batch["track_right"].to(device)
            waypoints = batch["waypoints"].to(device)
            labels_mask = batch["waypoints_mask"].to(device)

            # Forward pass
            outputs = model(track_left, track_right)
            loss = criterion(outputs * labels_mask[..., None], waypoints * labels_mask[..., None])
              # Scale loss by accumulation steps
            
            # Backward pass and gradient accumulation
            loss.backward()
            
            # Apply gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Accumulate gradients and update optimizer at intervals
            optimizer.step()
            optimizer.zero_grad()

            # Accumulate loss and update metrics
            total_loss += loss.item()
            if labels_mask.sum().item() > 0:
                train_metric.add(outputs, waypoints, labels_mask)
                processed_train_batches += 1

        # Compute training metrics
        train_metrics = train_metric.compute()
        avg_train_loss = total_loss / processed_train_batches if processed_train_batches > 0 else 0
        print(f"Training - Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_train_loss:.4f}, "
              f"Longitudinal Error: {train_metrics['longitudinal_error']:.4f}, "
              f"Lateral Error: {train_metrics['lateral_error']:.4f}, "
              f"L1 Error: {train_metrics['l1_error']:.4f}, "
              f"Number of Samples: {train_metrics['num_samples']}")

        # Validation phase
        model.eval()
        val_metric.reset()
        val_total_loss = 0.0
        processed_val_batches = 0  # Counter for batches with valid waypoints

        with torch.no_grad():
            for batch in val_loader:
                track_left = batch["track_left"].to(device)
                track_right = batch["track_right"].to(device)
                waypoints = batch["waypoints"].to(device)
                labels_mask = batch["waypoints_mask"].to(device)

                outputs = model(track_left, track_right)
                val_loss = criterion(outputs * labels_mask[..., None], waypoints * labels_mask[..., None])
                val_total_loss += val_loss.item()
                if labels_mask.sum().item() > 0:
                    val_metric.add(outputs, waypoints, labels_mask)
                    processed_val_batches += 1

        # Compute validation metrics
        val_metrics = val_metric.compute()
        avg_val_loss = val_total_loss / processed_val_batches if processed_val_batches > 0 else 0
        print(f"Validation - Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_val_loss:.4f}, "
              f"Longitudinal Error: {val_metrics['longitudinal_error']:.4f}, "
              f"Lateral Error: {val_metrics['lateral_error']:.4f}, "
              f"L1 Error: {val_metrics['l1_error']:.4f}, "
              f"Number of Samples: {val_metrics['num_samples']}")

        # Step the scheduler
        scheduler.step(avg_val_loss)
    
        if val_metrics["lateral_error"]<LAT_ERROR_TARGET and val_metrics["longitudinal_error"]<LON_ERROR_TARGET:
         break

    # Save the model at the end of training
    save_model(model)
    print("Training complete.")

if __name__ == "__main__":
    train_mlp_planner()

