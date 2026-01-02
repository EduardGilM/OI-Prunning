import torch
from transformers import ViTForImageClassification
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_cifar100_accuracy():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    base_dir = Path(__file__).parent.parent.parent
    model_path = base_dir / "ViT_CIFAR100" / "models" / "pytorch_model.bin"
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return

    print(f"Loading model from {model_path}")
    
    # Load model architecture
    model_name = "google/vit-base-patch16-224"
    model = ViTForImageClassification.from_pretrained(
        model_name, 
        num_labels=100, 
        ignore_mismatched_sizes=True
    )
    
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Data loading
    # ViT standard normalization for google/vit-base-patch16-224 is 0.5 mean/std
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    data_dir = base_dir / "ViT_CIFAR100" / "data"
    test_dataset = CIFAR100(root=str(data_dir), train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    correct = 0
    total = 0
    
    print("Evaluating...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print("-" * 30)
    print(f"Test Accuracy: {accuracy:.2f}%")
    print("-" * 30)

if __name__ == "__main__":
    test_cifar100_accuracy()
