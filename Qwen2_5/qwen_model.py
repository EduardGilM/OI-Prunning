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
        
        # Preserve original model name for save/load cycle
        if hasattr(self, '_original_model_name'):
            new_wrapper._original_model_name = self._original_model_name
        else:
            new_wrapper._original_model_name = self.model_name
        
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
        """
        Save the model with updated config reflecting pruned dimensions.
        
        IMPORTANT: For pruned models, we need to update the config's intermediate_size
        to reflect the new (possibly varying) layer dimensions. Since standard Qwen
        config only supports a single intermediate_size, we save the actual per-layer
        dimensions in a custom field.
        """
        import os
        import json
        
        # Update config to reflect pruned dimensions
        current_dims = self.get_mlp_dimensions()
        original_intermediate = getattr(self.config, 'intermediate_size', current_dims[0])
        
        # Check if model has been pruned (dimensions differ from original)
        is_pruned = any(d != original_intermediate for d in current_dims)
        
        # If all layers have the same dimension, update intermediate_size
        if len(set(current_dims)) == 1:
            self.model.config.intermediate_size = current_dims[0]
        else:
            # For varying dimensions, save the minimum (for compatibility)
            # and store actual dimensions in custom config
            self.model.config.intermediate_size = min(current_dims)
        
        # Save custom attribute for per-layer dimensions
        self.model.config.pruned_layer_dims = current_dims
        
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
        # Also save a metadata file with pruning info
        # Determine the original model name (needed for loading)
        original_model_name = self.model_name
        # If model_name is a local path (from previous load), try to get the original
        if hasattr(self, '_original_model_name'):
            original_model_name = self._original_model_name
        elif os.path.exists(self.model_name):
            # It's a local path, check for existing metadata
            existing_meta = os.path.join(self.model_name, "pruning_metadata.json")
            if os.path.exists(existing_meta):
                with open(existing_meta, 'r') as f:
                    old_meta = json.load(f)
                    original_model_name = old_meta.get("original_model_name", self.model_name)
        
        metadata = {
            "is_pruned": is_pruned or any(d != original_intermediate for d in current_dims),
            "original_model_name": original_model_name,
            "original_intermediate_size": original_intermediate,
            "pruned_layer_dims": current_dims,
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
        }
        metadata_path = os.path.join(path, "pruning_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(cls, path: str, device: str = None) -> 'QwenWrapper':
        """
        Load a saved model, handling pruned models correctly.
        
        For pruned models with per-layer varying dimensions, we need to:
        1. Load the base model from the original architecture
        2. Prune it to match the saved dimensions
        3. Load the pruned weights
        """
        import os
        import json
        from safetensors.torch import load_file as load_safetensors
        from glob import glob
        
        wrapper = cls.__new__(cls)
        wrapper.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Check if this is a pruned model by looking for metadata
        metadata_path = os.path.join(path, "pruning_metadata.json")
        is_pruned = os.path.exists(metadata_path)
        
        if is_pruned:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            pruned_dims = metadata.get("pruned_layer_dims", [])
            original_model_name = metadata.get("original_model_name", "HuggingFaceTB/SmolLM2-135M")
            
            # Load the base model first (with original architecture)
            base_model = AutoModelForCausalLM.from_pretrained(
                original_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            
            # Get the config for dimensions
            config = base_model.config
            original_intermediate_size = config.intermediate_size
            num_layers = config.num_hidden_layers
            
            # Create keep masks based on pruned dimensions
            layer_keep_masks = []
            for i in range(num_layers):
                if i < len(pruned_dims):
                    new_dim = pruned_dims[i]
                else:
                    new_dim = original_intermediate_size
                
                # Create mask: keep first new_dim neurons
                mask = torch.zeros(original_intermediate_size, dtype=torch.bool)
                mask[:new_dim] = True
                layer_keep_masks.append(mask)
            
            # Prune the base model to match saved dimensions
            for i, (layer, mask) in enumerate(zip(base_model.model.layers, layer_keep_masks)):
                indices = torch.where(mask)[0]
                new_dim = len(indices)
                
                if new_dim == original_intermediate_size:
                    continue  # No pruning needed for this layer
                
                old_gate = layer.mlp.gate_proj
                old_up = layer.mlp.up_proj
                old_down = layer.mlp.down_proj
                
                layer_device = old_gate.weight.device
                
                new_gate = nn.Linear(old_gate.in_features, new_dim, bias=old_gate.bias is not None)
                new_up = nn.Linear(old_up.in_features, new_dim, bias=old_up.bias is not None)
                new_down = nn.Linear(new_dim, old_down.out_features, bias=old_down.bias is not None)
                
                layer.mlp.gate_proj = new_gate.to(layer_device).half()
                layer.mlp.up_proj = new_up.to(layer_device).half()
                layer.mlp.down_proj = new_down.to(layer_device).half()
            
            # Now load the saved weights
            # Try safetensors first, then pytorch
            safetensor_files = glob(os.path.join(path, "*.safetensors"))
            if safetensor_files:
                state_dict = {}
                for sf in safetensor_files:
                    state_dict.update(load_safetensors(sf))
            else:
                # Fall back to pytorch bin files
                bin_files = glob(os.path.join(path, "*.bin"))
                state_dict = {}
                for bf in bin_files:
                    state_dict.update(torch.load(bf, map_location="cpu"))
            
            # Load state dict with strict=False to handle any remaining mismatches
            missing, unexpected = base_model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"Warning: Missing keys when loading pruned model: {len(missing)} keys")
            
            wrapper.model = base_model
            # Preserve original model name for future saves
            wrapper._original_model_name = original_model_name
        else:
            # Standard loading for non-pruned models
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
