import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from typing import Optional, Callable, Any
import os
from functools import wraps


def get_num_gpus() -> int:
    return torch.cuda.device_count()


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    if is_distributed():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    if is_distributed():
        return dist.get_world_size()
    return 1


def get_local_rank() -> int:
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    return 0


def get_device() -> torch.device:
    if torch.cuda.is_available():
        local_rank = get_local_rank()
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def is_main_process() -> bool:
    return get_rank() == 0


def init_distributed_mode(backend: str = "nccl") -> None:
    if not torch.cuda.is_available():
        return
    
    if dist.is_available() and dist.is_initialized():
        return
    
    num_gpus = get_num_gpus()
    if num_gpus < 2:
        return
    
    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = str(num_gpus)
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(int(os.environ.get("RANK", 0)) % num_gpus)
    
    local_rank = get_local_rank()
    torch.cuda.set_device(local_rank)
    
    dist.init_process_group(backend=backend)
    
    if is_main_process():
        print(f"Distributed training initialized with {get_world_size()} processes on {num_gpus} GPUs")


def cleanup_distributed_mode() -> None:
    if is_distributed():
        dist.destroy_process_group()


def setup_distributed_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size: int = 32,
    num_workers: int = 0,
) -> tuple:
    num_gpus = get_num_gpus()
    use_distributed = num_gpus > 1
    
    if use_distributed:
        init_distributed_mode()
    
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=get_world_size(),
        rank=get_rank(),
        shuffle=True,
        seed=42
    ) if use_distributed else None
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=get_world_size(),
        rank=get_rank(),
        shuffle=False,
        seed=42
    ) if use_distributed else None
    
    test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=get_world_size(),
        rank=get_rank(),
        shuffle=False,
        seed=42
    ) if use_distributed else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader, train_sampler


def wrap_model_distributed(model: torch.nn.Module) -> torch.nn.Module:
    num_gpus = get_num_gpus()
    
    if num_gpus > 1:
        init_distributed_mode()
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[get_local_rank()],
            output_device=get_local_rank(),
            find_unused_parameters=False
        )
        if is_main_process():
            print(f"Model wrapped with DistributedDataParallel on {num_gpus} GPUs")
    
    return model


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    return model


def print_once(msg: str, rank: int = 0) -> None:
    if get_rank() == rank:
        print(msg)


def synchronize_between_processes() -> None:
    if is_distributed():
        # Specify device_ids to avoid warnings and potential guessing issues
        local_rank = get_local_rank()
        dist.barrier(device_ids=[local_rank])
