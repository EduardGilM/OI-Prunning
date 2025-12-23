#!/usr/bin/env python3
import torch
import subprocess
import sys
import os
from pathlib import Path


def launch_distributed(script: str, script_args: list = None, num_gpus: int = None) -> int:
    if script_args is None:
        script_args = []
    
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    
    if num_gpus == 0:
        print("No GPUs detected. Running on CPU.")
        return subprocess.call([sys.executable, script] + script_args)
    
    if num_gpus == 1:
        print("Only 1 GPU detected. Running single GPU training.")
        return subprocess.call([sys.executable, script] + script_args)
    
    print(f"Detected {num_gpus} GPUs. Launching distributed training...")
    
    cmd = [
        sys.executable,
        "-m", "torch.distributed.launch",
        "--nproc_per_node", str(num_gpus),
        "--master_addr", "127.0.0.1",
        "--master_port", "29500",
        script
    ] + script_args
    
    return subprocess.call(cmd)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python launch_distributed.py <script> [args...]")
        sys.exit(1)
    
    script = sys.argv[1]
    script_args = sys.argv[2:] if len(sys.argv) > 2 else []
    
    if not Path(script).exists():
        print(f"Error: Script '{script}' not found.")
        sys.exit(1)
    
    exit_code = launch_distributed(script, script_args)
    sys.exit(exit_code)
