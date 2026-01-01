import torch
from torch import nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
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
from utils.oinfo import calculate_oinfo_gradient_distributed, calculate_oinfo_gradient


def get_layerwise_activations(model, val_loader, device, max_samples=1000):
    """
    Devuelve activaciones FFN por capa (lista de (name, tensor[N, hidden])).
    Se promedian los tokens por muestra (mean en dim=1) y se recorta a max_samples.
    """
    model.eval()
    accumulated = {}
    total_samples = 0

    with torch.no_grad():
        for batch_idx, (imgs, _) in enumerate(val_loader):
            imgs = imgs.to(device)
            batch_acts = model.get_layer_ffn_activations(imgs)

            if not accumulated:
                for name in batch_acts.keys():
                    accumulated[name] = []

            for name, act in batch_acts.items():
                act_flat = act.mean(dim=1).cpu()
                accumulated[name].append(act_flat)

            total_samples += imgs.size(0)
            if total_samples >= max_samples:
                break

    layer_tensors = {}
    for name, act_list in accumulated.items():
        layer_tensors[name] = torch.cat(act_list, dim=0)[:max_samples]

    sorted_keys = sorted(layer_tensors.keys(), key=lambda x: int(x.split('_')[-1]))
    layer_activations = []
    for name in sorted_keys:
        layer_activations.append((name, layer_tensors[name]))
    return layer_activations


def visualize_epoch_oinfo(layer_grads_list, iteration, output_dir):
    iter_dir = output_dir / f"iter_{iteration}"
    os.makedirs(iter_dir, exist_ok=True)
    
    # Guardar gradientes por capa
    np.savez(iter_dir / "grads.npz", **{name: grads for name, grads in layer_grads_list})
    
    for name, layer_grads in layer_grads_list:
        plt.figure(figsize=(10, 5))
        colors = ['red' if g > 0 else 'blue' for g in layer_grads]
        plt.bar(range(len(layer_grads)), layer_grads, color=colors, edgecolor='black', linewidth=0.5)
        plt.axhline(0, color='black', linewidth=0.8)
        plt.title(f"O-Info Gradients: {name} (Iter {iteration})")
        plt.xlabel("Neuron Index")
        plt.ylabel("Gradient (Synergy/Redundancy)")
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.savefig(iter_dir / f"{name}_gradients.png")
        plt.close()

def prune_vit_global():
    print("=" * 60)
    print("TEST: ViT FFN PRUNING")
    print("=" * 60)
    
    device = get_device()
    base_dir = Path(__file__).parent.parent
    plot_dir = base_dir / "plots" / "oinfo_vit"
    output_dir = base_dir / "plots" / "oi_pruning_vit"
    model_dir = base_dir / "ViT_MNIST" / "models"
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    baseline_path = model_dir / "vit_mnist_base.pth"
    pruned_path = model_dir / "vit_mnist_pruned.pth"
    
    train_loader, val_loader, _, _ = get_mnist_dataloaders(
        root=str(base_dir / "ViT_MNIST" / "data"), batch_size=64
    )

    # Cargar modelo: si hay pruned, continuar desde ahí; si no, usar baseline
    if pruned_path.exists():
        print(f"Cargando modelo prunado desde {pruned_path}")
        state = torch.load(pruned_path, map_location="cpu")

        # Inferir hidden_dims por capa a partir de los pesos prunados
        hidden_dims = []
        layer_idx = 0
        while f"encoder_blocks.{layer_idx}.linear1.weight" in state:
            w = state[f"encoder_blocks.{layer_idx}.linear1.weight"]
            hidden_dims.append(w.shape[0])
            layer_idx += 1

        print(f"Dimensiones ocultas prunadas detectadas: {hidden_dims}")

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

    else:
        if not baseline_path.exists():
            raise FileNotFoundError(
                f"El modelo base no se encontró en {baseline_path}. "
                "Por favor, asegúrate de que el modelo base existe antes de ejecutar el pruning."
            )
        print(f"Cargando modelo base desde {baseline_path}")
        model = create_model(device)
        model.load_state_dict(torch.load(baseline_path, map_location=device, weights_only=False))
        
    criterion = nn.CrossEntropyLoss()
    _, accuracy = evaluate(model, val_loader, criterion, device)
    print(f"Baseline Accuracy: {accuracy:.4f}")
    
    history = {
        'iteration': [0],
        'accuracy': [accuracy],
        'params': [sum(p.numel() for p in model.parameters())]
    }
    
    current_model = model
    iteration = 0
    
    while True:
        iteration += 1
        print(f"\n--- Pruning Iteration {iteration} ---")
        
        print("Collecting activations por capa...")
        layer_activations = get_layerwise_activations(
            current_model, val_loader, device, max_samples=1000
        )
        
        layer_grads_list = []
        layer_stds_list = []
        total_prunable = 0
        
        for name, act_tensor in layer_activations:
            act_np = act_tensor.numpy()
            stds = np.std(act_np, axis=0)
            layer_stds_list.append(stds)
            valid_indices = np.where(stds > 1e-6)[0]
            
            print(f"  {name}: {len(valid_indices)} neuronas activas de {act_np.shape[1]}")
            
            if len(valid_indices) > 0:
                X_active = act_np[:, valid_indices]
                print(f"    Calculando O-Info (GPU) para {name} ...")
                grads_active, o_val = calculate_oinfo_gradient(X_active, k=3)
                grads = np.zeros(act_np.shape[1])
                grads[valid_indices] = grads_active
                print(f"    O-Info {name}: {o_val:.4f}")
            else:
                print(f"    Sin neuronas activas en {name}.")
                grads = np.zeros(act_np.shape[1])
            
            layer_grads_list.append((name, grads))
            total_prunable += np.sum(grads > 0)
        
        visualize_epoch_oinfo(layer_grads_list, iteration, plot_dir)
        
        if total_prunable == 0:
            print("No redundant neurons found. Stopping.")
            break
            
        new_hidden_dims = []
        layer_keep_masks = []
        
        for (name, grads), stds in zip(layer_grads_list, layer_stds_list):
            is_active = stds > 1e-6
            is_redundant = grads > 0
            keep_mask = is_active & (~is_redundant)
            
            if np.sum(keep_mask) == 0:
                print(f"Warning: Layer {name} would be empty. Keeping 1 neuron.")
                keep_mask[np.argmax(stds)] = True
                
            new_dim = np.sum(keep_mask)
            new_hidden_dims.append(int(new_dim))
            layer_keep_masks.append(keep_mask)
            
            print(f"  Layer {name}: {len(grads)} -> {new_dim}")
            
        print("Building pruned model...")
        cfg = current_model.config
        new_model = ViT(
            num_patches=cfg['num_patches'],
            img_size=cfg['img_size'],
            num_classes=cfg['num_classes'],
            patch_size=cfg['patch_size'],
            embed_dim=cfg['embed_dim'],
            num_encoders=cfg['num_encoders'],
            num_heads=cfg['num_heads'],
            hidden_dim=cfg['hidden_dim'],
            dropout=cfg['dropout'],
            activation=cfg['activation'],
            in_channels=cfg['in_channels'],
            hidden_dims=new_hidden_dims,
        ).to(device)
        
        old_layers = current_model.encoder_blocks
        new_layers = new_model.encoder_blocks
        
        for i, (old_layer, new_layer) in enumerate(zip(old_layers, new_layers)):
            mask = layer_keep_masks[i]
            indices = np.where(mask)[0]
            
            new_layer.self_attn.load_state_dict(old_layer.self_attn.state_dict())
            new_layer.norm1.load_state_dict(old_layer.norm1.state_dict())
            new_layer.norm2.load_state_dict(old_layer.norm2.state_dict())
            
            with torch.no_grad():
                new_layer.linear1.weight.copy_(old_layer.linear1.weight[indices, :])
                new_layer.linear1.bias.copy_(old_layer.linear1.bias[indices])
                new_layer.linear2.weight.copy_(old_layer.linear2.weight[:, indices])
                new_layer.linear2.bias.copy_(old_layer.linear2.bias)
        
        new_model.embeddings_block.load_state_dict(current_model.embeddings_block.state_dict())
        new_model.mlp_head.load_state_dict(current_model.mlp_head.state_dict())
        
        current_model = new_model
        
        print("Fine-tuning (1 epoch)...")
        train_model(current_model, train_loader, val_loader, epochs=1, device=device)
        
        _, acc = evaluate(current_model, val_loader, criterion, device)
        params = sum(p.numel() for p in current_model.parameters())
        print(f"Iteration {iteration} Result: Acc={acc:.4f}, Params={params}")
        
        history['iteration'].append(iteration)
        history['accuracy'].append(acc)
        history['params'].append(params)
        
    print("Plotting compression results...")
    plt.figure(figsize=(10, 6))
    
    plt.plot(history['iteration'], history['accuracy'], marker='o', label='Accuracy')
    
    for i, txt in enumerate(history['params']):
        plt.annotate(f"{txt:,}", 
                     (history['iteration'][i], history['accuracy'][i]),
                     textcoords="offset points", 
                     xytext=(0,10), 
                     ha='center',
                     fontsize=9,
                     rotation=0)
                     
    plt.xlabel('Pruning Iteration')
    plt.ylabel('Test Accuracy')
    plt.title('ViT Pruning')
    plt.grid(True)
    plt.savefig(output_dir / "compression_vs_accuracy.png")
    print(f"Compression plot saved to {output_dir}")
    
    # Guardar el modelo podado final
    final_model_path = model_dir / "vit_mnist_pruned.pth"
    torch.save(current_model.state_dict(), final_model_path)
    final_params = sum(p.numel() for p in current_model.parameters())
    final_acc = history['accuracy'][-1]
    print(f"\nModelo podado guardado en: {final_model_path}")
    print(f"Parámetros finales: {final_params:,}")
    print(f"Precisión final: {final_acc:.4f}")

if __name__ == "__main__":
    prune_vit_global()
