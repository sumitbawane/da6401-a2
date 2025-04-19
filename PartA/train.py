import torch
import torch.nn as nn
import wandb
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from cnn import SimpleCNN
import torch.amp as amp

from data_utils import get_transforms, load_and_split_dataset
from model_utils import run_epoch

# Define data paths
data_path = '../inaturalist_12K/train'
data_path_test = '../inaturalist_12K/val'

def train():
    # Initialize wandb
    wandb.init()
    config = wandb.config  # Access sweep parameters
    
    # Set device (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = amp.GradScaler(enabled=device.type == 'cuda')
    
    # Set a meaningful name for the run
    wandb.run.name = (
        f"base_filter_size_{config.base_filter}_"
        f"filter_strategy_{config.filter_strategy}_"
        f"kernel_{'_'.join(map(str, config.kernel_sizes))}_"
        f"activation_{config.activation}_batchnorm_{config.batch_norm}_"
        f"dropout_{config.dropout}_fcsize_{config.fc_size}_epochs_{config.epochs}_"
        f"augmentation_{config.data_augmentation}_lr_{config.learning_rate:.1e}_"
        f"batchsize_{config.batch_size}"
    )
    
    # Configure filter sizes based on strategy
    base_filter = config.base_filter  # Base filter size for the first layer
    filters = []
    if config['filter_strategy'] == 'same':
        filters = [base_filter] * 5
    elif config['filter_strategy'] == 'doubling':
        filters = [base_filter * (2 ** i) for i in range(5)]
    elif config['filter_strategy'] == 'halving':
        filters = [base_filter * (2 ** i) for i in reversed(range(5))]

    # Extract parameters from wandb config
    no_kernels = filters
    kernel_size = config.kernel_sizes
    activation = config.activation
    batch_norm = config.batch_norm
    dropout = config.dropout
    fc_size = config.fc_size
    learning_rate = config.learning_rate
    batch_size = config.batch_size
    data_augmentation = config.data_augmentation
    epochs = config.epochs
    val_split = 0.2  # 20% of data for validation

    # Initialize the model
    model = SimpleCNN(
        num_classes=10,
        kernel_size=kernel_size,
        no_kernels=no_kernels,
        fc1_size=fc_size,
        conv_activation=activation,
        use_batch_norm=batch_norm,
        dropout=dropout
    )
    
    model = model.to(device)  # Move model to the appropriate device
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    if config.optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    elif config.optimizer_type == 'Nadam':
        optimizer = optim.NAdam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Load and split dataset
    train_loader, val_loader = load_and_split_dataset(
        data_path, 
        batch_size, 
        val_split, 
        data_augmentation
    )

    # Training loop
    best_val_accuracy = 0
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, optimizer, device, scaler, is_training=True)
        
        # Validation phase
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation for validation
            val_loss, val_accuracy = run_epoch(model, val_loader, criterion, optimizer, device, scaler, is_training=False)
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), f"best_model_{wandb.run.id}.pth")

        # Log metrics to wandb
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_accuracy,
            'val_loss': val_loss,
            'val_acc': val_accuracy
        })
        
        print(f"Epoch [{epoch + 1}/{epochs}], "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
    
    # Log final best validation accuracy and hyperparameters
    wandb.log({
        'best_val_acc': best_val_accuracy,
        'batch_size': batch_size,
        'activation': config.activation,
        'batch_norm': config.batch_norm,
        'dropout': config.dropout,
        'fc_size': config.fc_size,
        'learning_rate': config.learning_rate,
        'data_augmentation': config.data_augmentation,
        'epochs': config.epochs,
        'base_filter': config.base_filter,
        'filter_strategy': config.filter_strategy,
        'kernel_sizes': config.kernel_sizes
    })
    
    # Update the sweep metric
    wandb.run.summary["val_acc"] = best_val_accuracy