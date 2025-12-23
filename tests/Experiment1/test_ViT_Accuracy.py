import torch
import sys
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ViT_MNIST.vit_model import ViT
from ViT_MNIST.utils_vit import (
    create_model,
    get_mnist_dataloaders,
    evaluate,
    train_model,
    get_device,
    NUM_PATCHES,
    IMG_SIZE,
    NUM_CLASSES,
    PATCH_SIZE,
    EMBED_DIM,
    NUM_HEADS,
    HIDDEN_DIM,
    DROPOUT,
    ACTIVATION,
    IN_CHANNELS,
)

def test_vit_accuracy():
    print("=" * 60)
    print("TEST: ViT ACCURACY ON MNIST")
    print("=" * 60)
    
    device = get_device()
    print(f"Device: {device}")
    
    # Check for pruned model
    base_dir = Path(__file__).parent.parent
    model_dir = base_dir / "ViT_MNIST" / "models"
    model_path = model_dir / "vit_mnist_pruned.pth"
    os.makedirs(model_dir, exist_ok=True)
    
    train_loader, val_loader, test_loader = get_mnist_dataloaders(root=str(base_dir / "ViT_MNIST" / "data"))
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo podado en {model_path}. "
            "Ejecuta primero el script de pruning para generar 'vit_mnist_pruned.pth'."
        )

    print(f"Loading pruned model from {model_path}")
    state = torch.load(model_path, map_location="cpu")

    # Inferir hidden_dims por capa a partir de los pesos prunados
    hidden_dims = []
    layer_idx = 0
    while f"encoder_blocks.{layer_idx}.linear1.weight" in state:
        w = state[f"encoder_blocks.{layer_idx}.linear1.weight"]
        hidden_dims.append(w.shape[0])
        layer_idx += 1

    print(f"Detected pruned hidden_dims per layer: {hidden_dims}")

    model = ViT(
        num_patches=NUM_PATCHES,
        img_size=IMG_SIZE,
        num_classes=NUM_CLASSES,
        patch_size=PATCH_SIZE,
        embed_dim=EMBED_DIM,
        num_encoders=len(hidden_dims),
        num_heads=NUM_HEADS,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
        activation=ACTIVATION,
        in_channels=IN_CHANNELS,
        hidden_dims=hidden_dims,
    ).to(device)
    model.load_state_dict(state, strict=True)
        
    # Evaluate on Test Set
    print(f"\nEvaluating on Test Set...")
    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    
    print("-" * 30)
    print(f"Test Loss (Error): {test_loss:.4f}")
    print(f"Test Accuracy:     {test_acc*100:.2f}%")
    print("-" * 30)
    
    # Logic for confusion matrix
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for img, label in test_loader:
            img = img.to(device)
            output = model(img)
            preds = torch.argmax(output, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.numpy())
            
    # Plot Confusion Matrix
    output_dir = base_dir / "plots" / "vit_accuracy"
    os.makedirs(output_dir, exist_ok=True)
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix (Acc: {test_acc:.4f})')
    plt.savefig(output_dir / "confusion_matrix_prunned.png")
    print(f"Confusion matrix saved to {output_dir}")
    plt.close()
    
    return True

if __name__ == "__main__":
    success = test_vit_accuracy()
    sys.exit(0 if success else 1)
