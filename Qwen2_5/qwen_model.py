import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Dict, List, Optional, Tuple
import copy

class QwenWrapper:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str = None,
        load_in_4bit: bool = False,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, fix_mistral_regex=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.config = self.model.config
        self.num_layers = self.config.num_hidden_layers
        self.intermediate_size = self.config.intermediate_size
        self.hidden_size = self.config.hidden_size

    def get_layer_mlp_activations(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        max_samples: int = 1000
    ) -> Dict[str, torch.Tensor]:
        activations = {}
        hooks = []
        
        def get_hook(name):
            def hook(module, input, output):
                act = output.detach().float()
                if name in activations:
                    activations[name] = torch.cat([activations[name], act.mean(dim=1).cpu()], dim=0)
                else:
                    activations[name] = act.mean(dim=1).cpu()
            return hook
        
        for i, layer in enumerate(self.model.model.layers):
            hooks.append(layer.mlp.gate_proj.register_forward_hook(get_hook(f"layer_{i}")))
        
        self.model.eval()
        with torch.no_grad():
            self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        for h in hooks:
            h.remove()
            
        for name in activations:
            if activations[name].shape[0] > max_samples:
                activations[name] = activations[name][:max_samples]
                
        return activations

    def get_mlp_dimensions(self) -> List[int]:
        dims = []
        for layer in self.model.model.layers:
            dims.append(layer.mlp.gate_proj.out_features)
        return dims

    def prune_mlp_neurons(
        self,
        layer_keep_masks: List[torch.Tensor]
    ) -> 'QwenWrapper':
        if len(layer_keep_masks) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} masks, got {len(layer_keep_masks)}")
        
        new_wrapper = QwenWrapper.__new__(QwenWrapper)
        new_wrapper.model_name = self.model_name
        new_wrapper.device = self.device
        new_wrapper.tokenizer = self.tokenizer
        new_wrapper.config = copy.deepcopy(self.config)
        new_wrapper.num_layers = self.num_layers
        new_wrapper.hidden_size = self.hidden_size
        
        new_wrapper.model = copy.deepcopy(self.model)
        
        for i, (layer, mask) in enumerate(zip(new_wrapper.model.model.layers, layer_keep_masks)):
            indices = torch.where(mask)[0]
            new_dim = len(indices)
            
            if new_dim == 0:
                indices = torch.tensor([0])
                new_dim = 1
            
            old_gate = layer.mlp.gate_proj
            old_up = layer.mlp.up_proj
            old_down = layer.mlp.down_proj
            
            layer_device = old_gate.weight.device
            
            new_gate = nn.Linear(old_gate.in_features, new_dim, bias=old_gate.bias is not None)
            new_gate.weight.data = old_gate.weight.data[indices, :].clone()
            if old_gate.bias is not None:
                new_gate.bias.data = old_gate.bias.data[indices].clone()
            
            new_up = nn.Linear(old_up.in_features, new_dim, bias=old_up.bias is not None)
            new_up.weight.data = old_up.weight.data[indices, :].clone()
            if old_up.bias is not None:
                new_up.bias.data = old_up.bias.data[indices].clone()
            
            new_down = nn.Linear(new_dim, old_down.out_features, bias=old_down.bias is not None)
            new_down.weight.data = old_down.weight.data[:, indices].clone()
            if old_down.bias is not None:
                new_down.bias.data = old_down.bias.data.clone()
            
            layer.mlp.gate_proj = new_gate.to(layer_device).half()
            layer.mlp.up_proj = new_up.to(layer_device).half()
            layer.mlp.down_proj = new_down.to(layer_device).half()
        
        new_wrapper.intermediate_size = new_wrapper.get_mlp_dimensions()
        return new_wrapper

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def save(self, path: str):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @classmethod
    def load(cls, path: str, device: str = None) -> 'QwenWrapper':
        wrapper = cls.__new__(cls)
        wrapper.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        wrapper.model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        wrapper.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, fix_mistral_regex=True)
        if wrapper.tokenizer.pad_token is None:
            wrapper.tokenizer.pad_token = wrapper.tokenizer.eos_token
        wrapper.config = wrapper.model.config
        wrapper.num_layers = wrapper.config.num_hidden_layers
        wrapper.intermediate_size = wrapper.get_mlp_dimensions()
        wrapper.hidden_size = wrapper.config.hidden_size
        wrapper.model_name = path
        return wrapper
