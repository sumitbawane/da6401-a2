import torch
import torch.amp as amp

def run_epoch(model, dataloader, criterion, optimizer, device, scaler, is_training=True):
    """Run one epoch of training or validation."""
    total_loss = 0
    total = 0
    correct = 0
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass with autocast for mixed precision
        with amp.autocast(device_type=device.type, enabled=scaler.is_enabled() and is_training):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        # Backward pass and optimization (only during training)
        if is_training:
            optimizer.zero_grad()
            # Use scaler for mixed precision gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Accumulate loss
        total_loss += loss.item() * images.size(0)
    
    # Calculate average loss and accuracy
    avg_loss = total_loss / total
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy