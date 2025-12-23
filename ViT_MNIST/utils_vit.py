import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm
import timeit
from .vit_model import ViT

# Configuration Constants (Defaults)
RANDOM_SEED = 42
BATCH_SIZE = 32
NUM_CLASSES = 10
PATCH_SIZE = 4
IMG_SIZE = 28
IN_CHANNELS = 1
NUM_HEADS = 8
DROPOUT = 0.001
HIDDEN_DIM = 768
ADAM_WEIGHT_DECAY = 0
ADAM_BETAS = (0.9, 0.999)
ACTIVATION = "gelu"
NUM_ENCODERS = 4
EMBED_DIM = (PATCH_SIZE ** 2) * IN_CHANNELS
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def get_mnist_dataloaders(batch_size=32, root='.'):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    train_dataset = MNIST(root=root, train=True, download=True, transform=transform)
    test_dataset = MNIST(root=root, train=False, download=True, transform=transform)
    
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(RANDOM_SEED))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def create_model(device=None):
    if device is None:
        device = get_device()
        
    model = ViT(
        num_patches=NUM_PATCHES,
        img_size=IMG_SIZE,
        num_classes=NUM_CLASSES,
        patch_size=PATCH_SIZE,
        embed_dim=EMBED_DIM,
        num_encoders=NUM_ENCODERS,
        num_heads=NUM_HEADS,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
        activation=ACTIVATION,
        in_channels=IN_CHANNELS
    ).to(device)
    
    return model

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, leave=False, desc="Training")
    for img, label in pbar:
        img, label = img.to(device), label.to(device)
        
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Accuracy
        preds = torch.argmax(output, dim=1)
        correct += (preds == label).sum().item()
        total += label.size(0)
        
        pbar.set_postfix({'loss': running_loss/total, 'acc': correct/total})
        
    return running_loss / len(train_loader), correct / total

def evaluate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for img, label in tqdm(val_loader, leave=False, desc="Evaluating"):
            img, label = img.to(device), label.to(device)
            output = model(img)
            
            if criterion:
                loss = criterion(output, label)
                running_loss += loss.item()
            
            preds = torch.argmax(output, dim=1)
            correct += (preds == label).sum().item()
            total += label.size(0)
            
    avg_loss = running_loss / len(val_loader) if criterion else 0
    acc = correct / total
    return avg_loss, acc

def train_model(model, train_loader, val_loader, epochs=1, device=None):
    if device is None:
        device = get_device()
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), betas=ADAM_BETAS, lr=1e-4, weight_decay=ADAM_WEIGHT_DECAY)
    
    print(f"Training on {device} for {epochs} epochs...")
    
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.4f} Acc={train_acc:.4f} | Val Loss={val_loss:.4f} Acc={val_acc:.4f}")
        
    return model
