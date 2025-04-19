import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import models
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from PartA.train import run_epoch
from PartA.data_utils import get_transforms 
import wandb
data_path = '../inaturalist_12K/train'
# data_path_test = '../inaturalist_12K/val' 

# Init W&B
#wandb.init(project="finetune_inaturalist", name="efficientnetv2_finetune")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load dataset
dataset = datasets.ImageFolder(data_path, transform=get_transforms(data_augmentation=False,is_train=True))
val_size = int(0.2 * len(dataset))
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
val_ds.dataset.transform = get_transforms(data_augmentation=False,is_train=False)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4)

# Load EfficientNetV2 pre-trained
model = models.efficientnet_v2_s(weights="EfficientNet_V2_S_Weights.DEFAULT")

# Modify final layer
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 10)

# Freeze all layers first
for param in model.parameters():
    param.requires_grad = False

# Strategy: unfreeze last N layers
N = 20
for param in list(model.parameters())[-N:]:
    param.requires_grad = True

model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

# Training + Validation loop
def run_epoch(model, dataloader, train=False):
    model.train() if train else model.eval()
    running_loss, correct, total = 0.0, 0, 0
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        if train:
            optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        if train:
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
    return running_loss / total, 100 * correct / total

# Training loop
best_val_acc = 0
for epoch in range(10):
    train_loss, train_acc = run_epoch(model, train_loader, train=True)
    val_loss, val_acc = run_epoch(model, val_loader, train=False)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_efficientnetv2.pth")
    
    print(f"Epoch {epoch+1}: Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")

#wandb.run.summary["best_val_acc"] = best_val_acc
