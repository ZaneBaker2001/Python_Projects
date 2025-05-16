import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from homework.datasets.road_dataset import load_data
from homework.models import TransformerPlanner, save_model
from homework.metrics import PlannerMetric

def train_transformer_planner():
    # Hyperparameters
    train_dataset_path = "/Users/zanebaker/Desktop/Trajectory_Planner/drive_data/train"
    val_dataset_path = "/Users/zanebaker/Desktop/Trajectory_Planner/drive_data/val"
    hardware_batch_size = 10
    desired_batch_size = 160
    gradient_accumulation_steps = desired_batch_size // hardware_batch_size
    num_epochs = 50
    learning_rate = 1e-5  # Reduced for stability
     

    target_lon_error = 0.16
    target_lat_error = 0.5550

    # Load dataset
    train_loader = load_data(train_dataset_path, batch_size=hardware_batch_size, transform_pipeline="aug")
    val_loader = load_data(val_dataset_path, batch_size=hardware_batch_size, transform_pipeline="aug")

    # Initialize model
    model = TransformerPlanner(n_waypoints=3, d_model=128, nhead=2, num_decoder_layers=2, num_encoder_layers=2, dropout=0.1)
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)


    # Initialize metrics
    train_metric = PlannerMetric()
    val_metric = PlannerMetric()

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_metric.reset()
        total_loss = 0.0
        processed_train_batches = 0

        optimizer.zero_grad()
        for batch_idx, batch in enumerate(train_loader):
            track_left = batch["track_left"].to(device)
            track_right = batch["track_right"].to(device)
            waypoints = batch["waypoints"].to(device)
            labels_mask = batch["waypoints_mask"].to(device)

            outputs = model(track_left, track_right)
            loss = criterion(outputs * labels_mask[..., None], waypoints * labels_mask[..., None])
            loss = loss / gradient_accumulation_steps
            loss.backward()

            # Gradient clipping
            

            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * gradient_accumulation_steps
            if labels_mask.sum().item() > 0:
                train_metric.add(outputs, waypoints, labels_mask)
                processed_train_batches += 1

        # Compute training metrics
        train_metrics = train_metric.compute()
        avg_train_loss = total_loss / processed_train_batches if processed_train_batches > 0 else 0
        print(f"Training - Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_train_loss:.4f}, Longitudinal Error: {train_metrics['longitudinal_error']:.4f}, Lateral Error: {train_metrics['lateral_error']:.4f}, L1 Error: {train_metrics['l1_error']:.4f}")

        # Validation phase
        model.eval()
        val_metric.reset()
        val_total_loss = 0.0
        processed_val_batches = 0

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
        scheduler.step(avg_val_loss)
        print(f"Validation - Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_val_loss:.4f}, Longitudinal Error: {val_metrics['longitudinal_error']:.4f}, Lateral Error: {val_metrics['lateral_error']:.4f}, L1 Error: {val_metrics['l1_error']:.4f}")

        

        # Early stopping
        

        # Break if target errors are achieved
        if val_metrics["lateral_error"] < target_lat_error and val_metrics["longitudinal_error"] < target_lon_error:
            print("Target metrics achieved. Stopping training.")
            break

    save_model(model)
    print("Training complete.")

if __name__ == "__main__":
    train_transformer_planner()



