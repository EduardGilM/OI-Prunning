import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Qwen2_5.qwen_model import QwenWrapper
from Qwen2_5.utils_qwen import (
    get_calibration_data,
    fine_tune_lora,
    evaluate_with_harness,
    get_device,
    BENCHMARKS,
)
from utils.oinfo import calculate_oinfo_gradient_distributed, calculate_oinfo_gradient


def get_layerwise_activations(wrapper, max_samples=1000):
    print("Obteniendo datos de calibración...")
    input_ids, attention_mask = get_calibration_data(
        wrapper.tokenizer, 
        num_samples=max_samples
    )
    
    device = wrapper.device
    batch_size = 4
    
    all_activations = {}
    
    for i in range(0, input_ids.shape[0], batch_size):
        batch_ids = input_ids[i:i+batch_size].to(device)
        batch_mask = attention_mask[i:i+batch_size].to(device)
        
        batch_acts = wrapper.get_layer_mlp_activations(batch_ids, batch_mask)
        
        for name, act in batch_acts.items():
            if name in all_activations:
                all_activations[name] = torch.cat([all_activations[name], act], dim=0)
            else:
                all_activations[name] = act
    
    sorted_keys = sorted(all_activations.keys(), key=lambda x: int(x.split('_')[-1]))
    layer_activations = []
    for name in sorted_keys:
        tensor = all_activations[name][:max_samples]
        layer_activations.append((name, tensor))
    
    return layer_activations


def visualize_epoch_oinfo(layer_grads_list, iteration, output_dir):
    iter_dir = output_dir / f"iter_{iteration}"
    os.makedirs(iter_dir, exist_ok=True)
    
    np.savez(iter_dir / "grads.npz", **{name: grads for name, grads in layer_grads_list})
    
    for name, layer_grads in layer_grads_list:
        plt.figure(figsize=(14, 5))
        colors = ['red' if g > 0 else 'blue' for g in layer_grads]
        
        step = max(1, len(layer_grads) // 500)
        indices = range(0, len(layer_grads), step)
        sampled_grads = [layer_grads[i] for i in indices]
        sampled_colors = [colors[i] for i in indices]
        
        plt.bar(range(len(sampled_grads)), sampled_grads, color=sampled_colors, width=1.0)
        plt.axhline(0, color='black', linewidth=0.8)
        plt.title(f"O-Info Gradients: {name} (Iter {iteration})")
        plt.xlabel("Neuron Index")
        plt.ylabel("Gradient (Synergy/Redundancy)")
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(iter_dir / f"{name}_gradients.png", dpi=100)
        plt.close()


def prune_qwen_global():
    print("=" * 60)
    print("EXPERIMENT: Qwen2.5-1.5B MLP PRUNING with O-Information")
    print("=" * 60)
    
    device = get_device()
    base_dir = Path(__file__).parent.parent
    plot_dir = base_dir / "plots" / "oinfo_qwen"
    output_dir = base_dir / "plots" / "oi_pruning_qwen"
    model_dir = base_dir / "Qwen2.5" / "models"
    
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    pruned_path = model_dir / "qwen_pruned"
    
    if pruned_path.exists():
        print(f"Cargando modelo prunado desde {pruned_path}")
        wrapper = QwenWrapper.load(str(pruned_path))
    else:
        print("Cargando modelo base desde HuggingFace...")
        wrapper = QwenWrapper(
            model_name="Qwen/Qwen2.5-1.5B-Instruct",
            load_in_4bit=False,
        )
    
    print(f"Parámetros iniciales: {wrapper.count_parameters():,}")
    print(f"Número de capas: {wrapper.num_layers}")
    print(f"Dimensiones MLP: {wrapper.get_mlp_dimensions()[:3]}... (mostrando primeras 3)")
    
    print("\n--- Evaluación Baseline ---")
    baseline_results = evaluate_with_harness(wrapper, batch_size=4)
    print("Resultados baseline:")
    for benchmark, score in baseline_results.items():
        if score is not None:
            print(f"  {benchmark}: {score:.4f}")
        else:
            print(f"  {benchmark}: Error")
    
    history = {
        'iteration': [0],
        'params': [wrapper.count_parameters()],
        'benchmarks': [baseline_results],
    }
    
    iteration = 0
    max_iterations = 10
    
    while iteration < max_iterations:
        iteration += 1
        print(f"\n{'='*60}")
        print(f"--- Pruning Iteration {iteration} ---")
        print(f"{'='*60}")
        
        print("Recolectando activaciones por capa...")
        layer_activations = get_layerwise_activations(wrapper, max_samples=500)
        
        layer_grads_list = []
        layer_stds_list = []
        total_prunable = 0
        
        for name, act_tensor in tqdm(layer_activations, desc="Calculando O-Info"):
            act_np = act_tensor.numpy()
            stds = np.std(act_np, axis=0)
            layer_stds_list.append(stds)
            valid_indices = np.where(stds > 1e-6)[0]
            
            print(f"  {name}: {len(valid_indices)} neuronas activas de {act_np.shape[1]}")
            
            if len(valid_indices) > 0:
                max_neurons = min(len(valid_indices), 8960)
                if len(valid_indices) > max_neurons:
                    selected = np.random.choice(valid_indices, max_neurons, replace=False)
                    selected = np.sort(selected)
                else:
                    selected = valid_indices
                
                X_active = act_np[:, selected]
                grads_active, o_val = calculate_oinfo_gradient(X_active, k=3)
                
                grads = np.zeros(act_np.shape[1])
                grads[selected] = grads_active
                
                unselected = np.setdiff1d(valid_indices, selected)
                if len(unselected) > 0:
                    median_grad = np.median(grads_active)
                    grads[unselected] = median_grad
                
                print(f"    O-Info {name}: {o_val:.4f}")
            else:
                grads = np.zeros(act_np.shape[1])
            
            layer_grads_list.append((name, grads))
            total_prunable += np.sum(grads > 0)
        
        visualize_epoch_oinfo(layer_grads_list, iteration, plot_dir)
        
        if total_prunable == 0:
            print("No se encontraron neuronas redundantes. Finalizando.")
            break
        
        print(f"Total neuronas redundantes encontradas: {total_prunable}")
        
        layer_keep_masks = []
        new_dims = []
        
        for (name, grads), stds in zip(layer_grads_list, layer_stds_list):
            is_active = stds > 1e-6
            is_redundant = grads > 0
            keep_mask = is_active & (~is_redundant)
            
            min_neurons = 64
            if np.sum(keep_mask) < min_neurons:
                keep_indices = np.argsort(stds)[-min_neurons:]
                keep_mask = np.zeros_like(keep_mask, dtype=bool)
                keep_mask[keep_indices] = True
            
            new_dim = np.sum(keep_mask)
            new_dims.append(int(new_dim))
            layer_keep_masks.append(torch.tensor(keep_mask))
            
            print(f"  {name}: {len(grads)} -> {new_dim}")
        
        current_dims = wrapper.get_mlp_dimensions()
        if new_dims == current_dims:
            print("Sin cambios en dimensiones. Finalizando.")
            break
        
        print("\nConstruyendo modelo prunado...")
        wrapper = wrapper.prune_mlp_neurons(layer_keep_masks)
        
        print(f"Nuevos parámetros: {wrapper.count_parameters():,}")
        
        print("\nFine-tuning con LoRA (1 epoch)...")
        wrapper = fine_tune_lora(
            wrapper,
            epochs=1,
            batch_size=2,
            learning_rate=2e-4,
            gradient_accumulation_steps=8,
            max_samples=5000,
        )
        
        print("\nEvaluando en benchmarks...")
        iter_results = evaluate_with_harness(wrapper, batch_size=4)
        
        print(f"Resultados iteración {iteration}:")
        for benchmark, score in iter_results.items():
            if score is not None:
                print(f"  {benchmark}: {score:.4f}")
            else:
                print(f"  {benchmark}: Error")
        
        history['iteration'].append(iteration)
        history['params'].append(wrapper.count_parameters())
        history['benchmarks'].append(iter_results)
        
        wrapper.save(str(pruned_path))
    
    print("\n--- Generando gráficas de resumen ---")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    iterations = history['iteration']
    params = [p / 1e6 for p in history['params']]
    
    axes[0, 0].plot(iterations, params, 'bo-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Pruning Iteration')
    axes[0, 0].set_ylabel('Parameters (M)')
    axes[0, 0].set_title('Model Size vs Iteration')
    axes[0, 0].grid(True, alpha=0.3)
    
    benchmark_names = list(BENCHMARKS[0].keys()) if history['benchmarks'] else []
    if history['benchmarks']:
        benchmark_names = list(history['benchmarks'][0].keys())
    
    for idx, benchmark in enumerate(benchmark_names[:5]):
        row = (idx + 1) // 3
        col = (idx + 1) % 3
        
        scores = []
        for bench_results in history['benchmarks']:
            score = bench_results.get(benchmark)
            scores.append(score if score is not None else 0)
        
        axes[row, col].plot(iterations, scores, 'go-', linewidth=2, markersize=8)
        axes[row, col].set_xlabel('Pruning Iteration')
        axes[row, col].set_ylabel('Accuracy')
        axes[row, col].set_title(f'{benchmark}')
        axes[row, col].grid(True, alpha=0.3)
        axes[row, col].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_dir / "pruning_summary.png", dpi=150)
    plt.close()
    
    import json
    history_serializable = {
        'iteration': history['iteration'],
        'params': history['params'],
        'benchmarks': [
            {k: float(v) if v is not None else None for k, v in b.items()}
            for b in history['benchmarks']
        ]
    }
    with open(output_dir / "history.json", "w") as f:
        json.dump(history_serializable, f, indent=2)
    
    final_path = model_dir / "qwen_final_pruned"
    wrapper.save(str(final_path))
    
    print(f"\n{'='*60}")
    print("EXPERIMENTO COMPLETADO")
    print(f"{'='*60}")
    print(f"Modelo guardado en: {final_path}")
    print(f"Parámetros finales: {wrapper.count_parameters():,}")
    print(f"Reducción: {(1 - wrapper.count_parameters() / history['params'][0]) * 100:.1f}%")
    print(f"Gráficas guardadas en: {output_dir}")


if __name__ == "__main__":
    prune_qwen_global()
