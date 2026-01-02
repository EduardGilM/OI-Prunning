import os
from huggingface_hub import hf_hub_download
import sys
from pathlib import Path

def download_from_hf(repo_id, filename="pytorch_model.bin"):
    # Obtener la raíz del proyecto (un nivel arriba de este script)
    base_dir = Path(__file__).parent.parent
    local_dir = base_dir / "ViT_CIFAR100" / "models"
    
    print(f"Descargando {filename} desde el repo {repo_id}...")
    print(f"Destino: {local_dir}")
    
    os.makedirs(local_dir, exist_ok=True)
    
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False
    )
    
    print(f"Modelo descargado en: {path}")
    return path

if __name__ == "__main__":
    # Ejemplo de uso: python ViT_CIFAR100/download_hf_model.py "usuario/repo"
    if len(sys.argv) < 2:
        print("Uso: python ViT_CIFAR100/download_hf_model.py <REPO_ID> [FILENAME]")
        sys.exit(1)
    
    repo = sys.argv[1]
    fname = sys.argv[2] if len(sys.argv) > 2 else "pytorch_model.bin"
    
    try:
        download_from_hf(repo, fname)
    except Exception as e:
        print(f"Error: {e}")
