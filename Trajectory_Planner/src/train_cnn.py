import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from homework.datasets.road_dataset import load_data
from homework.models import CNNPlanner, save_model
from homework.metrics import PlannerMetric

def train_cnn():
    # Hyperparameters
    train_dataset_path = "/Users/zanebaker/Desktop/Trajectory_Planner/drive_data/train"
    val_dataset_path = "/Users/zanebaker/Desktop/Trajectory_Planner/drive_data/val"
    batch_size = 128
    num_epochs = 150  # Increased epochs for better convergence
    learning_rate = 1e-3  # Reduced learning rate for finer adjustments

    target_lat_value=0.44
    target_lon_value=0.31

    # Load dataset
    train_loader = load_data(train_dataset_path, batch_size=batch_size, transform_pipeline="default")  # Use augmentations
    val_loader = load_data(val_dataset_path, batch_size=batch_size, transform_pipeline="default")  # Use augmentations

    # Initialize model
    model = CNNPlanner(n_waypoints=3, input_height=96, input_width=128)
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    

    
    # Initialize metrics
    metrics = PlannerMetric()
    best_val_l1_error = float('inf')
    patience = 20  # Early stopping patience
    patience_counter = 0

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        metrics.reset()
        
        processed_train_batches = 0  # Counter for batches with valid waypoints
        for batch in train_loader:
            images = batch["image"].to(device)
            waypoints = batch["waypoints"].to(device)
            labels_mask = batch["waypoints_mask"].to(device)

            optimizer.zero_grad()  
            outputs = model(images) 

            # Calculate masked loss and update metrics
            loss = criterion(outputs * labels_mask[..., None], waypoints * labels_mask[..., None])
            loss.backward()

            # Gradient clipping
            

            optimizer.step()
            total_loss += loss.item()
            if labels_mask.sum().item() > 0:  # Ensure there are valid waypoints in the batch
                metrics.add(outputs, waypoints, labels_mask)
                processed_train_batches += 1

        # Compute training metrics
        train_metrics = metrics.compute()
        avg_train_loss = total_loss / processed_train_batches if processed_train_batches > 0 else 0
        
        print(f"Training - Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_train_loss:.4f}, Longitudinal Error: {train_metrics['longitudinal_error']:.4f}, Lateral Error: {train_metrics['lateral_error']:.4f}, L1 Error: {train_metrics['l1_error']:.4f}, Number of Samples: {train_metrics['num_samples']}")

        # Validation loop
        model.eval()
        val_total_loss = 0
        metrics.reset()
        processed_val_batches = 0  # Counter for batches with valid waypoints

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                waypoints = batch["waypoints"].to(device)
                labels_mask = batch["waypoints_mask"].to(device)
                outputs = model(images)
                
                val_loss = criterion(outputs * labels_mask[..., None], waypoints * labels_mask[..., None])
                val_total_loss += val_loss.item()

                if labels_mask.sum().item() > 0:  # Ensure there are valid waypoints in the batch
                    metrics.add(outputs, waypoints, labels_mask)
                    processed_val_batches += 1

        # Compute validation metrics
        val_metrics = metrics.compute()
        avg_val_loss = val_total_loss / processed_val_batches if processed_val_batches > 0 else 0
        scheduler.step()
        print(f"Validation - Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_val_loss:.4f}, Longitudinal Error: {val_metrics['longitudinal_error']:.4f}, Lateral Error: {val_metrics['lateral_error']:.4f}, L1 Error: {val_metrics['l1_error']:.4f}, Number of Samples: {val_metrics['num_samples']}")
        if val_metrics["lateral_error"]<target_lat_value and val_metrics["longitudinal_error"]<target_lon_value:
            break

        # Check for early stopping


        # Check if target errors are achieved
    

    # Save the model
    save_model(model)
    print("Training complete.")

if __name__ == "__main__":
    train_cnn()
