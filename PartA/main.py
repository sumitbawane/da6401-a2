import wandb
from train import train

# Configure WandB service wait time
wandb._service_wait = 60

sweep_config = {
    'method': 'bayes',  # Optimization method: 'grid', 'random', or 'bayes'
    'metric': {'name': 'val_acc', 'goal': 'maximize'},
    'parameters': {
        # strategy for filter size
        'filter_strategy': {'values': ['same', 'doubling', 'halving']},
        #base filter size
        'base_filter': {'values': [32, 64]},
        
        # Kernel sizes for each layer
        'kernel_sizes': {'values': [[3, 3, 5, 5, 7], [5, 5, 7, 7, 3], [3, 5, 7, 5, 3], [3, 3, 3, 3, 3], [5, 5, 5, 5, 5]]},
        'optimizer_type': {'values': ['Adam', 'Nadam']},
        # Activation function
        'activation': {'values': ['ReLU', 'GELU', 'SiLU', 'Mish']},

        # Batch normalization
        'batch_norm': {'values': [True, False]},

        # Dropout
        'dropout': {'values': [0.0, 0.2, 0.3]},

        # Fully connected layer size
        'fc_size': {'values': [256, 512]},

        # Learning rate
        'learning_rate': {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-3},

        # Batch size
        'batch_size': {'values': [32, 64]},

        # Data augmentation
        'data_augmentation': {'values': [True, False]},

        # Number of epochs
        'epochs': {'values': [10, 15, 20]}
    }
}

if __name__ == "__main__":
    wandb.login()
    sweep_id = wandb.sweep(sweep_config, project="da6401 a2")
    wandb.agent(sweep_id, train, count=1)