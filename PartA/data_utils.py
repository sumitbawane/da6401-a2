import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def get_transforms(data_augmentation, is_training=True):
    """Get the appropriate transforms based on whether we're training and using augmentation."""
    if is_training and data_augmentation:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def load_and_split_dataset(data_path, batch_size, val_split, data_augmentation):
    """Load and split the dataset into training and validation sets."""
    # Load dataset
    dataset = datasets.ImageFolder(
        root=data_path,
        transform=get_transforms(data_augmentation, is_training=True)
    )
    
    # Calculate split sizes
    dataset_size = len(dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    
    # Split the dataset
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    # Update validation set transforms (no augmentation for validation)
    val_dataset.dataset.transform = get_transforms(False, is_training=False)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        pin_memory=True, 
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,  # No need to shuffle validation data
        pin_memory=True, 
        num_workers=4
    )
    
    return train_loader, val_loader