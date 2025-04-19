import torch
import torch.optim as optim
import torch.amp as amp  # Import for mixed precision training
from cnn import SimpleCNN  # Import the SimpleCNN model
from data_utils import load_and_split_dataset  # Import the dataset loading and splitting function
from model_utils import run_epoch  # Import the function to run training/validation epochs

# Best configuration from the sweep
best_config = {
    'base_filter': 64,
    'filter_strategy': 'doubling',
    'kernel_sizes': [3, 5, 7, 5, 3],
    'activation': 'ReLU',
    'batch_norm': True,
    'dropout': 0.2,
    'fc_size': 512,
    'learning_rate': 3.3e-5,
    'batch_size': 32,
    'data_augmentation': False,
    'epochs': 15,
    'optimizer_type': 'Adam'
}

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize GradScaler for mixed precision training
scaler = amp.GradScaler(enabled=device.type == 'cuda')

# Determine filter sizes based on strategy
base_filter = best_config['base_filter']
if best_config['filter_strategy'] == 'same':
    filters = [base_filter] * 5
elif best_config['filter_strategy'] == 'doubling':
    filters = [base_filter * (2 ** i) for i in range(5)]
elif best_config['filter_strategy'] == 'halving':
    filters = [base_filter * (2 ** i) for i in reversed(range(5))]

# Initialize model with best config
model = SimpleCNN(
    num_classes=10,
    kernel_size=best_config['kernel_sizes'],
    no_kernels=filters,
    fc1_size=best_config['fc_size'],
    conv_activation=best_config['activation'],
    use_batch_norm=best_config['batch_norm'],
    dropout=best_config['dropout']
)
model = model.to(device)

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
if best_config['optimizer_type'] == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=best_config['learning_rate'], weight_decay=1e-4)
elif best_config['optimizer_type'] == 'Nadam':
    optimizer = optim.NAdam(model.parameters(), lr=best_config['learning_rate'], weight_decay=1e-4)

# Load and split dataset using the utility function
data_path = '../inaturalist_12K/train'  # Path to training data
train_loader, val_loader = load_and_split_dataset(
    data_path=data_path,
    batch_size=best_config['batch_size'],
    val_split=0.2,
    data_augmentation=best_config['data_augmentation']
)

# Train the model
def train_best_model():
    print("Starting training with best configuration...")
    best_val_accuracy = 0
    model_save_path = "best_model.pth"

    for epoch in range(best_config['epochs']):
        # Training phase
        model.train()
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, optimizer, device, scaler, is_training=True
        )
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            val_loss, val_accuracy = run_epoch(
                model, val_loader, criterion, optimizer, device, scaler, is_training=False
            )
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved with validation accuracy: {val_accuracy:.2f}%")
        
        print(f"Epoch [{epoch + 1}/{best_config['epochs']}], "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

    print(f"Training completed. Best validation accuracy: {best_val_accuracy:.2f}%")

if __name__ == "__main__":
    train_best_model()