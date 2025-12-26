import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import load_dataset
from tqdm.auto import tqdm
import subprocess
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.distributed import (
    get_device, get_num_gpus, is_main_process, init_distributed_mode,
    wrap_model_distributed, unwrap_model, print_once, get_rank, get_world_size,
    synchronize_between_processes
)

DOLLY_DATASET = "databricks/databricks-dolly-15k"
CALIBRATION_SAMPLES = 1000

LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": TaskType.CAUSAL_LM,
}

# Set to False to run full evaluation (takes hours)
QUICK_MODE = True

BENCHMARKS_QUICK = [
    {"name": "hellaswag", "num_fewshot": 10},
    {"name": "arc_challenge", "num_fewshot": 25},
]

BENCHMARKS_FULL = [
    {"name": "hellaswag", "num_fewshot": 10},
    {"name": "mmlu", "num_fewshot": 5},
    {"name": "arc_challenge", "num_fewshot": 25},
    {"name": "winogrande", "num_fewshot": 5},
    {"name": "global_mmlu_lite", "num_fewshot": 5},
]

BENCHMARKS = BENCHMARKS_QUICK if QUICK_MODE else BENCHMARKS_FULL


class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        instruction = item.get("instruction", "")
        context = item.get("context", "")
        response = item.get("response", "")
        
        if context:
            prompt = f"### Instruction:\n{instruction}\n\n### Context:\n{context}\n\n### Response:\n{response}"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
        
        encoded = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze(),
            "labels": encoded["input_ids"].squeeze(),
        }


def load_dolly_dataset(tokenizer, max_samples: int = None, max_length: int = 512):
    dataset = load_dataset(DOLLY_DATASET, split="train")
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    return InstructionDataset(dataset, tokenizer, max_length)


def get_calibration_data(tokenizer, num_samples: int = CALIBRATION_SAMPLES) -> Tuple[torch.Tensor, torch.Tensor]:
    dataset = load_dataset(DOLLY_DATASET, split="train")
    dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    prompts = []
    for item in dataset:
        instruction = item.get("instruction", "")
        context = item.get("context", "")
        if context:
            prompt = f"### Instruction:\n{instruction}\n\n### Context:\n{context}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        prompts.append(prompt)
    
    encoded = tokenizer(
        prompts,
        truncation=True,
        max_length=256,
        padding=True,
        return_tensors="pt"
    )
    
    return encoded["input_ids"], encoded["attention_mask"]


def setup_lora(model, config: dict = None):
    if config is None:
        config = LORA_CONFIG
    
    lora_config = LoraConfig(**config)
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model


def fine_tune_lora(
    wrapper,
    epochs: int = 1,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    gradient_accumulation_steps: int = 4,
    max_samples: int = None,
):
    model = wrapper.model
    tokenizer = wrapper.tokenizer
    device = wrapper.device
    
    use_distributed = get_num_gpus() > 1
    if use_distributed:
        init_distributed_mode()
    
    peft_model = setup_lora(model)
    peft_model.train()
    
    train_dataset = load_dolly_dataset(tokenizer, max_samples=max_samples)
    
    if use_distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=True,
            seed=42
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=0,
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        train_sampler = None
    
    optimizer = torch.optim.AdamW(peft_model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs // gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )
    
    global_step = 0
    for epoch in range(epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}") if is_main_process() else train_loader
        
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = peft_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()
            total_loss += loss.item()
            
            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(peft_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
            
            if is_main_process() and isinstance(pbar, tqdm):
                pbar.set_postfix({"loss": total_loss / (step + 1)})
    
    merged_model = peft_model.merge_and_unload()
    wrapper.model = merged_model
    
    return wrapper


def evaluate_benchmarks(
    model_path: str,
    benchmarks: List[dict] = None,
    batch_size: int = 4,
    output_path: str = None,
) -> Dict[str, float]:
    if benchmarks is None:
        benchmarks = BENCHMARKS
    
    results = {}
    
    for benchmark in benchmarks:
        name = benchmark["name"]
        num_fewshot = benchmark.get("num_fewshot", 0)
        
        base_cmd = ["lm_eval"]
        args = [
            "--model", "hf",
            "--model_args", f"pretrained={model_path},trust_remote_code=True",
            "--tasks", name,
            "--num_fewshot", str(num_fewshot),
            "--batch_size", str(batch_size),
            "--output_path", output_path or "eval_results",
        ]
        
        cmd = base_cmd + args
        
        try:
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            except FileNotFoundError:
                # Try python -m lm_eval
                cmd = [sys.executable, "-m", "lm_eval"] + args
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

            if result.returncode == 0:
                # Parse del output para obtener accuracy
                output_lines = result.stdout.split("\n")
                for line in output_lines:
                    if "acc" in line.lower() and name in line.lower():
                        parts = line.split("|")
                        for part in parts:
                            if "." in part:
                                try:
                                    score = float(part.strip())
                                    if 0 <= score <= 1:
                                        results[name] = score
                                        break
                                except ValueError:
                                    continue
                
                if name not in results:
                    results[name] = None
                    print(f"Could not parse results for {name}")
            else:
                print(f"Error evaluating {name}: {result.stderr}")
                if "No such file or directory" in str(result.stderr) or result.returncode == 127:
                     print("Make sure lm-evaluation-harness is installed: pip install lm-evaluation-harness")
                results[name] = None
                
        except subprocess.TimeoutExpired:
            print(f"Timeout evaluating {name}")
            results[name] = None
        except FileNotFoundError:
             print(f"Command 'lm_eval' not found. Please install it: pip install lm-evaluation-harness")
             results[name] = None
        except Exception as e:
            print(f"Exception evaluating {name}: {e}")
            results[name] = None
    
    return results


def evaluate_with_harness(
    wrapper,
    benchmarks: List[dict] = None,
    batch_size: int = 16,
) -> Dict[str, float]:
    try:
        from lm_eval import evaluator
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        print("lm-evaluation-harness not installed. Please install it with: pip install lm-eval")
        return {}
        
    import shutil
    
    if benchmarks is None:
        benchmarks = BENCHMARKS
    
    temp_path = "tmp_eval_model"
    
    if is_main_process():
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)
        wrapper.save(temp_path)
    
    synchronize_between_processes()
    
    try:
        if is_main_process():
            print(f"Cargando modelo para evaluacion desde {temp_path}...", flush=True)
        
        lm = HFLM(
            pretrained=temp_path,
            batch_size=batch_size,
            trust_remote_code=True,
        )
        
        task_names = [b["name"] for b in benchmarks]
        num_fewshot_map = {b["name"]: b.get("num_fewshot", 0) for b in benchmarks}
        
        results = {}
        
        for task in task_names:
            try:
                if is_main_process():
                    print(f"Evaluando benchmark: {task} (num_fewshot={num_fewshot_map[task]})...", flush=True)
                
                eval_results = evaluator.simple_evaluate(
                    model=lm,
                    tasks=[task],
                    num_fewshot=num_fewshot_map[task],
                    batch_size=batch_size,
                )
                
                if eval_results and "results" in eval_results:
                    task_results = eval_results["results"].get(task, {})
                    acc = task_results.get("acc,none") or task_results.get("acc_norm,none") or task_results.get("acc")
                    results[task] = acc
                    if is_main_process():
                        print(f"  {task}: {acc}", flush=True)
                else:
                    results[task] = None
                
            except Exception as e:
                if is_main_process():
                    print(f"Error evaluating {task}: {e}", flush=True)
                results[task] = None
                
    finally:
        synchronize_between_processes()
        if is_main_process() and os.path.exists(temp_path):
            try:
                shutil.rmtree(temp_path)
            except:
                pass
    
    return results
