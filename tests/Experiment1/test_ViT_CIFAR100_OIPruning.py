import torch
from torch import nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
from tqdm import tqdm
from transformers import ViTForImageClassification

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ViT_CIFAR100.utils_vit import (
    get_cifar100_dataloaders,
    evaluate,
    train_model,
    get_device,
    NUM_CLASSES,
    IMG_SIZE,
)
from ViT_CIFAR100.download_hf_model import download_from_hf
from utils.oinfo import calculate_oinfo_gradient, calculate_oinfo_gradient_distributed
from utils.distributed import (
    init_distributed_mode,
    wrap_model_distributed,
    is_main_process,
    synchronize_between_processes,
    unwrap_model
)


def get_layerwise_activations(model, val_loader, device, max_samples=500):
    """
    Devuelve activaciones de la capa intermedia (FFN) por cada bloque del encoder.
    """
    model.eval()
    accumulated = {}
    total_samples = 0
    
    hooks = []
    activations = {}

    def get_hook(name):
        def hook(module, input, output):
            # output shape: [batch, seq_len, intermediate_size]
            # Promediamos sobre los tokens (dim=1) para tener un vector por muestra
            activations[name] = output.detach().mean(dim=1).cpu()
        return hook

    # Registrar hooks en las capas intermediate.dense
    for i, layer in enumerate(model.vit.encoder.layer):
        name = f"encoder_{i}"
        hooks.append(layer.intermediate.dense.register_forward_hook(get_hook(name)))

    with torch.no_grad():
        for imgs, _ in val_loader:
            imgs = imgs.to(device)
            model(imgs)

            if not accumulated:
                for name in activations.keys():
                    accumulated[name] = []

            for name, act in activations.items():
                accumulated[name].append(act)

            total_samples += imgs.size(0)
            if total_samples >= max_samples:
                break

    # Limpiar hooks
    for hook in hooks:
        hook.remove()

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
        plt.bar(range(len(layer_grads)), layer_grads, color=colors, edgecolor='black', linewidth=0.1)
        plt.axhline(0, color='black', linewidth=0.8)
        plt.title(f"O-Info Gradients: {name} (Iter {iteration})")
        plt.xlabel("Neuron Index")
        plt.ylabel("Gradient (Synergy/Redundancy)")
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.savefig(iter_dir / f"{name}_gradients.png")
        plt.close()

def prune_transformers_model(model, layer_keep_masks):
    """
    Modifica el modelo in-place reemplazando las capas Linear por versiones reducidas.
    """
    for i, mask in enumerate(layer_keep_masks):
        layer = model.vit.encoder.layer[i]
        indices = np.where(mask)[0]
        
        old_inter = layer.intermediate.dense
        old_out = layer.output.dense
        
        new_inter_size = len(indices)
        
        # Crear nuevas capas
        new_inter = nn.Linear(old_inter.in_features, new_inter_size).to(old_inter.weight.device)
        new_out = nn.Linear(new_inter_size, old_out.out_features).to(old_out.weight.device)
        
        with torch.no_grad():
            new_inter.weight.copy_(old_inter.weight[indices, :])
            new_inter.bias.copy_(old_inter.bias[indices])
            new_out.weight.copy_(old_out.weight[:, indices])
            new_out.bias.copy_(old_out.bias)
            
        # Reemplazar
        layer.intermediate.dense = new_inter
        layer.output.dense = new_out
        
    return model

def prune_vit_cifar100():
    init_distributed_mode()
    
    if is_main_process():
        print("=" * 60)
        print("TEST: ViT CIFAR100 FFN PRUNING (DISTRIBUTED)")
        print("=" * 60)
    
    device = get_device()
    base_dir = Path(__file__).parent.parent.parent
    plot_dir = base_dir / "plots" / "Experiment1.2" / "oinfo_vit_cifar100"
    output_dir = base_dir / "plots" / "Experiment1.2" / "oi_pruning_vit_cifar100"
    model_dir = base_dir / "ViT_CIFAR100" / "models"
    
    if is_main_process():
        os.makedirs(plot_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
    
    model_path = model_dir / "pytorch_model.bin"
    pruned_path = model_dir / "pytorch_model_pruned.bin"
    
    if not model_path.exists() and is_main_process():
        print(f"Modelo no encontrado en {model_path}")
        repo_id = os.environ.get("HF_REPO_ID")
        if repo_id:
            print(f"Intentando descargar desde Hugging Face: {repo_id}")
            download_from_hf(repo_id)
        else:
            print("Error: El modelo no existe y no se ha definido HF_REPO_ID en las variables de entorno.")
            print("Usa: export HF_REPO_ID='tu_usuario/tu_repo' antes de ejecutar.")
            sys.exit(1)
    
    synchronize_between_processes()
    
    train_loader, val_loader, _, _ = get_cifar100_dataloaders(
        root=str(base_dir / "ViT_CIFAR100" / "data"), batch_size=32
    )

    if is_main_process():
        print(f"Cargando modelo desde {model_path}")
    
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224", 
        num_labels=100, 
        ignore_mismatched_sizes=True
    )
    
    state_dict = torch.load(model_path, map_location="cpu")
    
    hidden_dims = []
    layer_idx = 0
    while f"vit.encoder.layer.{layer_idx}.intermediate.dense.weight" in state_dict:
        w = state_dict[f"vit.encoder.layer.{layer_idx}.intermediate.dense.weight"]
        hidden_dims.append(w.shape[0])
        layer_idx += 1
    
    if is_main_process():
        print(f"Dimensiones detectadas en el checkpoint: {hidden_dims}")
    
    for i, h_dim in enumerate(hidden_dims):
        if h_dim != 3072:
            layer = model.vit.encoder.layer[i]
            layer.intermediate.dense = nn.Linear(768, h_dim)
            layer.output.dense = nn.Linear(h_dim, 768)
            
    model.load_state_dict(state_dict)
    model.to(device)
    
    # Wrap model for DDP
    model = wrap_model_distributed(model)
        
    criterion = nn.CrossEntropyLoss()
    _, accuracy = evaluate(model, val_loader, criterion, device)
    
    if is_main_process():
        print(f"Initial Accuracy: {accuracy:.4f}")
    
    history = {
        'iteration': [0],
        'accuracy': [accuracy],
        'params': [sum(p.numel() for p in model.parameters())]
    }
    
    current_model = model
    iteration = 0
    MAX_ITERATIONS = 5
    
    while iteration < MAX_ITERATIONS:
        iteration += 1
        if is_main_process():
            print(f"\n--- Pruning Iteration {iteration} ---")
        
        if is_main_process():
            print("Collecting activations...")
        
        # get_layerwise_activations uses hooks, should work with DDP model
        layer_activations = get_layerwise_activations(
            current_model, val_loader, device, max_samples=500
        )
        
        layer_grads_list = []
        layer_stds_list = []
        total_prunable = 0
        
        for name, act_tensor in layer_activations:
            act_np = act_tensor.numpy()
            stds = np.std(act_np, axis=0)
            layer_stds_list.append(stds)
            valid_indices = np.where(stds > 1e-6)[0]
            
            if is_main_process():
                print(f"  {name}: {len(valid_indices)} neuronas activas de {act_np.shape[1]}")
            
            if len(valid_indices) > 0:
                X_active = act_np[:, valid_indices]
                # Use distributed O-Info calculation
                grads_active, o_val = calculate_oinfo_gradient_distributed(X_active, k=3)
                grads = np.zeros(act_np.shape[1])
                grads[valid_indices] = grads_active
                if is_main_process():
                    print(f"    O-Info {name}: {o_val:.4f}")
            else:
                if is_main_process():
                    print(f"    Sin neuronas activas en {name}.")
                grads = np.zeros(act_np.shape[1])
            
            layer_grads_list.append((name, grads))
            total_prunable += np.sum(grads > 0)
        
        if is_main_process():
            visualize_epoch_oinfo(layer_grads_list, iteration, plot_dir)
        
        if total_prunable == 0:
            if is_main_process():
                print("No redundant neurons found. Stopping.")
            break
            
        layer_keep_masks = []
        
        for (name, grads), stds in zip(layer_grads_list, layer_stds_list):
            is_active = stds > 1e-6
            is_redundant = grads > 0
            keep_mask = is_active & (~is_redundant)
            
            if np.sum(keep_mask) == 0:
                if is_main_process():
                    print(f"Warning: Layer {name} would be empty. Keeping 1 neuron.")
                keep_mask[np.argmax(stds)] = True
                
            layer_keep_masks.append(keep_mask)
            if is_main_process():
                print(f"  Layer {name}: {len(grads)} -> {np.sum(keep_mask)}")
            
        if is_main_process():
            print("Pruning model...")
        
        # Unwrap to modify architecture
        raw_model = unwrap_model(current_model)
        current_model = prune_transformers_model(raw_model, layer_keep_masks)
        
        # Re-wrap for DDP
        current_model = wrap_model_distributed(current_model)
        
        if is_main_process():
            print("Fine-tuning (1 epoch)...")
        
        train_model(current_model, train_loader, val_loader, epochs=1, device=device, lr=1e-5)
        
        _, acc = evaluate(current_model, val_loader, criterion, device)
        params = sum(p.numel() for p in current_model.parameters())
        
        if is_main_process():
            print(f"Iteration {iteration} Result: Acc={acc:.4f}, Params={params}")
            history['iteration'].append(iteration)
            history['accuracy'].append(acc)
            history['params'].append(params)
            torch.save(unwrap_model(current_model).state_dict(), pruned_path)
        
        synchronize_between_processes()
        
    if is_main_process():
        print("Plotting compression results...")
        plt.figure(figsize=(10, 6))
        plt.plot(history['iteration'], history['accuracy'], marker='o', label='Accuracy')
        for i, txt in enumerate(history['params']):
            plt.annotate(f"{txt:,}", (history['iteration'][i], history['accuracy'][i]), 
                         textcoords="offset points", xytext=(0,10), ha='center')
        plt.xlabel('Pruning Iteration')
        plt.ylabel('Test Accuracy')
        plt.title('ViT CIFAR100 Pruning')
        plt.grid(True)
        plt.savefig(output_dir / "compression_vs_accuracy.png")
        print(f"Final pruned model saved to {pruned_path}")

if __name__ == "__main__":
    prune_vit_cifar100()
