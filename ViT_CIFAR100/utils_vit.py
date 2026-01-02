import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR100
from tqdm import tqdm
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.distributed import (
    get_device, get_num_gpus, is_main_process, init_distributed_mode,
    wrap_model_distributed, unwrap_model, print_once, setup_distributed_dataloaders,
    synchronize_between_processes
)

# Configuration Constants for CIFAR100 (matching google/vit-base-patch16-224)
RANDOM_SEED = 42
BATCH_SIZE = 32
NUM_CLASSES = 100
PATCH_SIZE = 16
IMG_SIZE = 224
IN_CHANNELS = 3
NUM_HEADS = 12
DROPOUT = 0.0
HIDDEN_DIM = 3072 # This is the intermediate_size in transformers
EMBED_DIM = 768
NUM_ENCODERS = 12
ACTIVATION = "gelu"
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2

def get_cifar100_dataloaders(batch_size=32, root='.', use_distributed=None):
    if use_distributed is None:
        use_distributed = get_num_gpus() > 1
    
    # ViT standard normalization
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    train_dataset = CIFAR100(root=root, train=True, download=True, transform=transform)
    test_dataset = CIFAR100(root=root, train=False, download=True, transform=transform)
    
    # Split train into train and val (90/10)
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size], 
                                            generator=torch.Generator().manual_seed(RANDOM_SEED))
    
    if use_distributed:
        train_loader, val_loader, test_loader, _ = setup_distributed_dataloaders(
            train_subset, val_subset, test_dataset, batch_size=batch_size
        )
    else:
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
    return train_loader, val_loader, test_loader, None

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            if hasattr(outputs, 'logits'):
                outputs = outputs.logits
            
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # Synchronize results in distributed mode
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        stats = torch.tensor([running_loss, correct, total], device=device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        running_loss, correct, total = stats.tolist()
        # Adjust running_loss by world size for the average
        num_batches = len(dataloader)
        stats_batches = torch.tensor([float(num_batches)], device=device)
        dist.all_reduce(stats_batches, op=dist.ReduceOp.SUM)
        total_batches = stats_batches.item()
        return running_loss / total_batches, correct / total
            
    return running_loss / len(dataloader), correct / total

def train_model(model, train_loader, val_loader, epochs, device, lr=2e-5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        # Set epoch for sampler if it exists
        if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
            
        model.train()
        running_loss = 0.0
        
        # Only show progress bar on main process
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}") if is_main_process() else train_loader
        
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            if hasattr(outputs, 'logits'):
                outputs = outputs.logits
                
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        if is_main_process():
            print(f"Epoch {epoch+1} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Ensure all processes are synced before next epoch
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
