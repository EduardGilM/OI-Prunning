import torch
from torch import nn
import numpy as np
import matplotlib

matplotlib.use("Agg")
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
    get_mnist_dataloaders_train_test,
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
from ViT_MNIST.utils_vit import get_class_filtered_dataloader


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

    sorted_keys = sorted(layer_tensors.keys(), key=lambda x: int(x.split("_")[-1]))
    layer_activations = []
    for name in sorted_keys:
        layer_activations.append((name, layer_tensors[name]))
    return layer_activations


def get_layerwise_activations_per_class(model, class_loader, device, max_samples=1000):
    """
    Devuelve activaciones FFN por capa para una clase específica.

    Args:
        model: Modelo ViT
        class_loader: DataLoader filtrado por clase específica
        device: CUDA/CPU
        max_samples: Máximo de muestras a procesar

    Returns:
        layer_activations: Lista de (name, tensor[N, hidden])
    """
    model.eval()
    accumulated = {}
    total_samples = 0

    with torch.no_grad():
        for batch_idx, (imgs, _) in enumerate(class_loader):
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

    sorted_keys = sorted(layer_tensors.keys(), key=lambda x: int(x.split("_")[-1]))
    layer_activations = []
    for name in sorted_keys:
        layer_activations.append((name, layer_tensors[name]))
    return layer_activations


def calculate_oi_for_all_classes(
    model, train_dataset, device, batch_size=500, max_samples=1000
):
    """
    Calcula OI gradient para cada clase (0-9).

    Args:
        model: Modelo ViT
        train_dataset: Dataset de entrenamiento completo
        device: CUDA/CPU
        batch_size: Tamaño del batch (500 para cálculo de OI)
        max_samples: Máximo de muestras por clase

    Returns:
        dict: {class_id: {'layer_grads': [(name, grads), ...], 'layer_activations': [(name, tensor), ...]}}
    """
    class_oi_results = {}

    for class_id in range(10):
        print(f"\n{'=' * 60}")
        print(f"Calculando OI para clase {class_id}")
        print(f"{'=' * 60}")

        class_loader = get_class_filtered_dataloader(
            train_dataset, class_id, batch_size
        )

        layer_activations = get_layerwise_activations_per_class(
            model, class_loader, device, max_samples=max_samples
        )

        layer_grads_list = []
        for name, act_tensor in layer_activations:
            act_np = act_tensor.numpy()
            stds = np.std(act_np, axis=0)
            valid_indices = np.where(stds > 1e-6)[0]

            print(
                f"  {name}: {len(valid_indices)} neuronas activas de {act_np.shape[1]}"
            )

            if len(valid_indices) > 0:
                X_active = act_np[:, valid_indices]
                grads_active, o_val = calculate_oinfo_gradient(X_active, k=3)
                grads = np.zeros(act_np.shape[1])
                grads[valid_indices] = grads_active
                print(f"    O-Info {name}: {o_val:.4f}")
            else:
                grads = np.zeros(act_np.shape[1])
                print(f"    Sin neuronas activas en {name}")

            layer_grads_list.append((name, grads))

        class_oi_results[class_id] = {
            "layer_grads": layer_grads_list,
            "layer_activations": layer_activations,
        }

    return class_oi_results


def find_redundant_neurons_all_classes(class_oi_results):
    """
    Encuentra neuronas redundantes en TODAS las clases.

    Una neurona se poda solo si es redundante (grad > 0) en TODAS las clases simultáneamente.

    Args:
        class_oi_results: Diccionario con resultados de OI por clase

    Returns:
        layer_keep_masks: Lista de mascaras booleanas (True=keep, False=prune)
    """
    print(f"\n{'=' * 60}")
    print("Encontrando neuronas redundantes en todas las clases")
    print(f"{'=' * 60}")

    first_class = class_oi_results[0]
    layer_keep_masks = []

    for layer_idx, (layer_name, grads) in enumerate(first_class["layer_grads"]):
        keep_mask = np.ones(len(grads), dtype=bool)

        for class_id in range(10):
            _, class_grads = class_oi_results[class_id]["layer_grads"][layer_idx]

            is_redundant = class_grads > 0
            keep_mask = keep_mask & (~is_redundant)

        layer_keep_masks.append(keep_mask)

        neurons_to_prune = np.sum(~keep_mask)
        print(
            f"  {layer_name}: {neurons_to_prune} neuronas redundantes en todas las clases"
        )

    return layer_keep_masks


def visualize_epoch_oinfo(layer_grads_list, iteration, output_dir):
    iter_dir = output_dir / f"iter_{iteration}"
    os.makedirs(iter_dir, exist_ok=True)

    # Guardar gradientes por capa
    np.savez(
        iter_dir / "grads.npz", **{name: grads for name, grads in layer_grads_list}
    )

    for name, layer_grads in layer_grads_list:
        plt.figure(figsize=(10, 5))
        colors = ["red" if g > 0 else "blue" for g in layer_grads]
        plt.bar(
            range(len(layer_grads)),
            layer_grads,
            color=colors,
            edgecolor="black",
            linewidth=0.5,
        )
        plt.axhline(0, color="black", linewidth=0.8)
        plt.title(f"O-Info Gradients: {name} (Iter {iteration})")
        plt.xlabel("Neuron Index")
        plt.ylabel("Gradient (Synergy/Redundancy)")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.savefig(iter_dir / f"{name}_gradients.png")
        plt.close()


def visualize_per_class_oinfo(class_oi_results, iteration, output_dir):
    """
    Guarda gráficas de OI por clase y por iteración.

    Crea directorios para cada iteración y cada clase (0-9) con las gráficas correspondientes.

    Args:
        class_oi_results: Diccionario con resultados de OI por clase
        iteration: Número de iteración de pruning
        output_dir: Directorio base para guardar las gráficas
    """
    iter_dir = output_dir / f"iter_{iteration}"
    os.makedirs(iter_dir, exist_ok=True)

    for class_id in range(10):
        class_dir = iter_dir / f"class_{class_id}"
        os.makedirs(class_dir, exist_ok=True)

        layer_grads_list = class_oi_results[class_id]["layer_grads"]

        np.savez(
            class_dir / "grads.npz", **{name: grads for name, grads in layer_grads_list}
        )

        for name, layer_grads in layer_grads_list:
            plt.figure(figsize=(10, 5))
            colors = ["red" if g > 0 else "blue" for g in layer_grads]
            plt.bar(
                range(len(layer_grads)),
                layer_grads,
                color=colors,
                edgecolor="black",
                linewidth=0.5,
            )
            plt.axhline(0, color="black", linewidth=0.8)
            plt.title(f"O-Info Gradients: {name} (Class {class_id}, Iter {iteration})")
            plt.xlabel("Neuron Index")
            plt.ylabel("Gradient (Synergy/Redundancy)")
            plt.grid(axis="y", linestyle="--", alpha=0.5)
            plt.savefig(class_dir / f"{name}_gradients.png")
            plt.close()


def prune_vit_global():
    print("=" * 60)
    print("TEST: ViT FFN PRUNING - PER CLASS")
    print("=" * 60)

    device = get_device()
    base_dir = Path(__file__).parent.parent.parent
    plot_dir = base_dir / "plots" / "oinfo_vit"
    output_dir = base_dir / "plots" / "oi_pruning_vit_per_class"
    model_dir = base_dir / "ViT_MNIST" / "models"
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    baseline_path = model_dir / "vit_mnist_base.pth"
    pruned_path = model_dir / "vit_mnist_pruned.pth"

    # NUEVO: Solo train y test (sin val)
    train_loader, test_loader = get_mnist_dataloaders_train_test(
        root=str(base_dir / "ViT_MNIST" / "data"), batch_size=32
    )

    # Obtener dataset completo para filtrar por clase
    train_dataset = train_loader.dataset

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
        model.load_state_dict(
            torch.load(baseline_path, map_location=device, weights_only=False)
        )

    criterion = nn.CrossEntropyLoss()

    # NUEVO: Evaluar baseline en test (no val)
    _, accuracy = evaluate(model, test_loader, criterion, device)
    print(f"Baseline Test Accuracy: {accuracy:.4f}")

    history = {
        "iteration": [0],
        "accuracy_before_ft": [accuracy],
        "accuracy_after_ft": [accuracy],
        "params": [sum(p.numel() for p in model.parameters())],
    }

    current_model = model
    iteration = 0

    while True:
        iteration += 1
        print(f"\n{'=' * 60}")
        print(f"PRUNING ITERATION {iteration}")
        print(f"{'=' * 60}")

        # PASO 1: Calcular OI para todas las clases
        print("\n[PASO 1] Calculando OI para todas las clases (0-9)...")
        class_oi_results = calculate_oi_for_all_classes(
            current_model, train_dataset, device, batch_size=500, max_samples=1000
        )

        # PASO 2: Encontrar neuronas redundantes en TODAS las clases
        print("\n[PASO 2] Encontrando neuronas redundantes en todas las clases...")
        layer_keep_masks = find_redundant_neurons_all_classes(class_oi_results)

        # Verificar si hay neuronas para podar
        total_prunable = sum(np.sum(~mask) for mask in layer_keep_masks)
        if total_prunable == 0:
            print("\n✓ No redundant neurons found in all classes. Stopping.")
            break

        # PASO 3: Visualizar OI por clase
        print("\n[PASO 3] Guardando visualizaciones...")
        visualize_per_class_oinfo(class_oi_results, iteration, plot_dir)

        # PASO 4: Construir modelo podado
        print("\n[PASO 4] Construyendo modelo podado...")
        new_hidden_dims = [int(np.sum(mask)) for mask in layer_keep_masks]

        cfg = current_model.config
        new_model = ViT(
            num_patches=cfg["num_patches"],
            img_size=cfg["img_size"],
            num_classes=cfg["num_classes"],
            patch_size=cfg["patch_size"],
            embed_dim=cfg["embed_dim"],
            num_encoders=cfg["num_encoders"],
            num_heads=cfg["num_heads"],
            hidden_dim=cfg["hidden_dim"],
            dropout=cfg["dropout"],
            activation=cfg["activation"],
            in_channels=cfg["in_channels"],
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

        new_model.embeddings_block.load_state_dict(
            current_model.embeddings_block.state_dict()
        )
        new_model.mlp_head.load_state_dict(current_model.mlp_head.state_dict())

        # PASO 5: Evaluar modelo podado en test
        print("\n[PASO 5] Evaluando modelo podado en test...")
        _, acc_before_ft = evaluate(new_model, test_loader, criterion, device)
        print(f"  Accuracy antes de fine-tune: {acc_before_ft:.4f}")

        # PASO 6: Fine-tune con 5 epochs en todo el train set
        print("\n[PASO 6] Fine-tuning con 5 epochs en todo el train set...")
        train_model(new_model, train_loader, None, epochs=5, device=device)

        # PASO 7: Evaluar después de fine-tune
        print("\n[PASO 7] Evaluando modelo después de fine-tune...")
        _, acc_after_ft = evaluate(new_model, test_loader, criterion, device)
        print(f"  Accuracy después de fine-tune: {acc_after_ft:.4f}")

        # PASO 8: Guardar checkpoint final de iteración
        final_checkpoint_path = model_dir / f"vit_mnist_pruned_iter_{iteration}.pth"
        torch.save(
            {
                "iteration": iteration,
                "model_state_dict": new_model.state_dict(),
                "accuracy": acc_after_ft,
                "hidden_dims": new_hidden_dims,
            },
            final_checkpoint_path,
        )
        print(f"  Checkpoint guardado: {final_checkpoint_path}")

        # Actualizar modelo actual
        current_model = new_model

        # Guardar en historia
        history["iteration"].append(iteration)
        history["accuracy_before_ft"].append(acc_before_ft)
        history["accuracy_after_ft"].append(acc_after_ft)
        history["params"].append(sum(p.numel() for p in current_model.parameters()))

        print(f"\n{'=' * 60}")
        print(f"RESUMEN ITERACIÓN {iteration}")
        print(f"{'=' * 60}")
        print(f"  Parámetros: {history['params'][-1]:,}")
        print(f"  Acc antes de FT: {acc_before_ft:.4f}")
        print(f"  Acc después de FT: {acc_after_ft:.4f}")
        print(f"  Neuronas podadas: {total_prunable}")

    # Plotting compression results
    print("\nPlotting compression results...")
    plt.figure(figsize=(12, 6))

    plt.plot(
        history["iteration"],
        history["accuracy_before_ft"],
        marker="o",
        label="Accuracy antes FT",
        linestyle="--",
    )
    plt.plot(
        history["iteration"],
        history["accuracy_after_ft"],
        marker="s",
        label="Accuracy después FT",
    )

    for i, txt in enumerate(history["params"]):
        plt.annotate(
            f"{txt:,}",
            (history["iteration"][i], history["accuracy_after_ft"][i]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
            rotation=0,
        )

    plt.xlabel("Pruning Iteration")
    plt.ylabel("Test Accuracy")
    plt.title("ViT Pruning - Per Class (Redundant in ALL classes)")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "compression_vs_accuracy.png")
    print(f"Compression plot saved to {output_dir}")

    # Guardar el modelo podado final
    final_model_path = model_dir / "vit_mnist_pruned.pth"
    torch.save(current_model.state_dict(), final_model_path)
    final_params = sum(p.numel() for p in current_model.parameters())
    final_acc = history["accuracy_after_ft"][-1]
    print(f"\n{'=' * 60}")
    print(f"RESULTADO FINAL")
    print(f"{'=' * 60}")
    print(f"Modelo podado guardado en: {final_model_path}")
    print(f"Parámetros finales: {final_params:,}")
    print(f"Precisión final: {final_acc:.4f}")
    print(f"Iteraciones completadas: {iteration}")


if __name__ == "__main__":
    prune_vit_global()
