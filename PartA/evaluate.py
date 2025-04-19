import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from cnn import SimpleCNN  # Import the SimpleCNN model
import numpy as np
import numpy as np
from PIL import Image, ImageDraw, ImageFont
# Path to test data
test_data_path = '../inaturalist_12K/val'

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


# Define test data transforms
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the test dataset
test_dataset = datasets.ImageFolder(
    root=test_data_path,
    transform=test_transform
)

# Create test data loader
test_loader = DataLoader(
    test_dataset,
    batch_size=best_config['batch_size'],
    shuffle=False,
    pin_memory=True,
    num_workers=4
)

# Get class names
class_names = test_dataset.classes

# Determine filter sizes based on strategy
base_filter = best_config['base_filter']
if best_config['filter_strategy'] == 'same':
    filters = [base_filter] * 5
elif best_config['filter_strategy'] == 'doubling':
    filters = [base_filter * (2 ** i) for i in range(5)]
elif best_config['filter_strategy'] == 'halving':
    filters = [base_filter * (2 ** i) for i in reversed(range(5))]

# Initialize the model with the best configuration
model = SimpleCNN(
    num_classes=len(class_names),  # Number of classes in the dataset
    kernel_size=best_config['kernel_sizes'],
    no_kernels=filters,
    fc1_size=best_config['fc_size'],
    conv_activation=best_config['activation'],
    use_batch_norm=best_config['batch_norm'],
    dropout=best_config['dropout']
)
model = model.to(device)

# Load the best model weights
model_save_path = "best_model.pth"
model.load_state_dict(torch.load(model_save_path, map_location=device))
model.eval()

# Evaluate on test set
def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_images = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            # Store predictions, labels, and images for visualization
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            # Convert images to numpy for visualization (only store a subset if needed)
            for img in images.cpu().numpy():
                all_images.append(img)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy, all_preds, all_labels, all_images

def plot_grid(images, preds, labels, class_names, rows=10, cols=3):


    # Normalize stats
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    num_samples = rows * cols

    # Select correct/incorrect
    correct = [i for i, (p, l) in enumerate(zip(preds, labels)) if p == l]
    incorrect = [i for i, (p, l) in enumerate(zip(preds, labels)) if p != l]
    n_incorrect = min(num_samples // 3, len(incorrect))
    n_correct = num_samples - n_incorrect

    selected = list(np.random.choice(correct, n_correct, replace=False)) + \
               list(np.random.choice(incorrect, n_incorrect, replace=False))
    np.random.shuffle(selected)

    if len(selected) < num_samples:
        remaining = num_samples - len(selected)
        pool = [i for i in range(len(images)) if i not in selected]
        selected += list(np.random.choice(pool, remaining, replace=False))

    # Layout settings
    image_size = 180
    padding_x = 40
    padding_y = 30
    margin = 60
    text_height = 45
    cell_width = image_size + padding_x
    cell_height = image_size + text_height + padding_y

    grid_width = cols * cell_width + 2 * margin
    grid_height = rows * cell_height + 2 * margin + 80  # + title

    grid_image = Image.new('RGB', (grid_width, grid_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(grid_image)

    try:
        font = ImageFont.truetype("Arial.ttf", 18)
        small_font = ImageFont.truetype("Arial.ttf", 14)
    except IOError:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    # Title
    title = "Test Set Predictions (Best CNN)"
    title_x = (grid_width - draw.textlength(title, font=font)) // 2
    draw.text((title_x, 20), title, fill=(0, 0, 0), font=font)

    model_info = f"{best_config['filter_strategy'].capitalize()} filters | {best_config['activation']} | Dropout={best_config['dropout']}"
    draw.text((margin, 50), model_info, fill=(60, 60, 60), font=small_font)
    draw.text((margin, 70), f"Test Accuracy: {test_accuracy:.2f}%", fill=(60, 60, 60), font=small_font)

    # Draw each image block
    for idx, img_idx in enumerate(selected):
        img = images[img_idx].transpose(1, 2, 0) * std + mean
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)
        img_pil = Image.fromarray(img).resize((image_size, image_size), Image.BICUBIC)

        row, col = divmod(idx, cols)
        x = margin + col * cell_width
        y = margin + row * cell_height + 80

        pred = preds[img_idx]
        label = labels[img_idx]
        is_correct = pred == label

        bg_color = (235, 255, 235) if is_correct else (255, 235, 235)
        draw.rectangle([x - 8, y - 8, x + image_size + 8, y + image_size + text_height + 8], fill=bg_color, outline=(200, 200, 200))

        grid_image.paste(img_pil, (x, y))

        # Text
        draw.text((x, y + image_size + 5), f"True: {class_names[label]}", fill=(0, 100, 0) if is_correct else (150, 0, 0), font=small_font)
        draw.text((x, y + image_size + 22), f"Pred: {class_names[pred]}", fill=(0, 0, 100), font=small_font)

    return grid_image




# Run the evaluation
if __name__ == "__main__":
    test_accuracy, predictions, true_labels, test_images = evaluate_model(model, test_loader, device)
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    grid_image = plot_grid(test_images, predictions, true_labels, class_names)
    grid_image.save("prediction_grid.png")
    print("10×3 prediction grid saved!")
    
    
    


