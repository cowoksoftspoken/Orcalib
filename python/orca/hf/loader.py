import urllib.request
import json
import os
import numpy as np
from typing import Dict, Any
from safetensors.numpy import load_file
import orca
import orca.nn as nn
from orca.tensor import Tensor
from orca.zoo import ResNet18, BERT, GPT2

CACHE_DIR = os.path.expanduser("~/.cache/orca/hub")

def download_file(url: str, dest_path: str):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path):
        return
    print(f"Downloading {url} to {dest_path}...")
    
    def reporthook(block_num, block_size, total_size):
        read_so_far = block_num * block_size
        if total_size > 0:
            percent = min(100, read_so_far * 100 / total_size)
            print(f"\rProgress: {percent:.2f}% ({read_so_far / (1024*1024):.2f}MB / {total_size / (1024*1024):.2f}MB)", end="")
        else:
            print(f"\rDownloaded {read_so_far / (1024*1024):.2f}MB", end="")
            
    urllib.request.urlretrieve(url, dest_path, reporthook)
    print("\nDownload finished.")

def load(model_name: str, pretrained: bool = True) -> nn.Module:
    """
    Downloads/loads configuration and pretrained weights from Hugging Face Hub,
    instantiates the blueprint model zoo, and maps the parameters.
    """
    model_mapping = {
        "gpt2": {
            "repo": "gpt2",
            "class": GPT2,
            "config_fn": "config.json",
            "weights_fn": "model.safetensors"
        },
        "bert-base-uncased": {
            "repo": "bert-base-uncased",
            "class": BERT,
            "config_fn": "config.json",
            "weights_fn": "model.safetensors"
        },
        "resnet18": {
            "repo": "microsoft/resnet-18",
            "class": ResNet18,
            "config_fn": "config.json",
            "weights_fn": "model.safetensors"
        }
    }

    if model_name not in model_mapping:
        raise ValueError(f"Unknown model name '{model_name}'. Available: {list(model_mapping.keys())}")

    info = model_mapping[model_name]
    repo = info["repo"]
    
    # 1. Download config & weights
    config_url = f"https://huggingface.co/{repo}/resolve/main/{info['config_fn']}"
    weights_url = f"https://huggingface.co/{repo}/resolve/main/{info['weights_fn']}"
    
    config_path = os.path.join(CACHE_DIR, model_name, info["config_fn"])
    weights_path = os.path.join(CACHE_DIR, model_name, info["weights_fn"])
    
    if pretrained:
        download_file(config_url, config_path)
        download_file(weights_url, weights_path)
        
        with open(config_path, "r") as f:
            config = json.load(f)
            
        weights = load_file(weights_path)
    else:
        # Default architecture values
        config = {}
        weights = {}

    # 2. Instantiate corresponding model zoo blueprint
    if model_name == "gpt2":
        model = GPT2(
            vocab_size=config.get("vocab_size", 50257),
            max_position_embeddings=config.get("n_positions", 1024),
            embed_dim=config.get("n_embd", 768),
            num_heads=config.get("n_head", 12),
            num_layers=config.get("n_layer", 12)
        )
    elif model_name == "bert-base-uncased":
        model = BERT(
            vocab_size=config.get("vocab_size", 30522),
            max_position_embeddings=config.get("max_position_embeddings", 512),
            type_vocab_size=config.get("type_vocab_size", 2),
            hidden_size=config.get("hidden_size", 768),
            num_attention_heads=config.get("num_attention_heads", 12),
            num_hidden_layers=config.get("num_hidden_layers", 12)
        )
    else:  # resnet18
        model = ResNet18(num_classes=config.get("num_labels", 1000))

    if not pretrained:
        return model

    # 3. Key Mapping & Load weights
    print(f"Loading pretrained weights for {model_name}...")
    
    def copy_tensor(param: nn.Parameter, np_arr: np.ndarray, reshape_to=None):
        if reshape_to:
            np_arr = np_arr.reshape(reshape_to)
        # Check for NaN in np_arr
        has_nan = np.isnan(np_arr).any()
        if has_nan:
            print(f"      [copy_tensor WARNING] Input numpy array contains NaN!")
        flat = np_arr.flatten().tolist()
        shape = list(np_arr.shape)
        if not shape:
            shape = [1]
        param.tensor = Tensor.from_list(flat, shape=shape).to(str(param.tensor.device))
        # Verify copied values
        copied_list = param.tensor.to_list()
        copied_nan = any(x != x for x in copied_list)
        if copied_nan:
            print(f"      [copy_tensor WARNING] Copied Orca tensor contains NaN!")

    if model_name == "gpt2":
        # Map GPT-2
        copy_tensor(model.wte.weight, weights["wte.weight"])
        copy_tensor(model.wpe.weight, weights["wpe.weight"])
        copy_tensor(model.ln_f.weight, weights["ln_f.weight"])
        copy_tensor(model.ln_f.bias, weights["ln_f.bias"])
        
        num_layers = config.get("n_layer", 12)
        for i in range(num_layers):
            block = model.blocks._modules[str(i)].block
            
            copy_tensor(block.self_attn.in_proj.weight, weights[f"h.{i}.attn.c_attn.weight"])
            copy_tensor(block.self_attn.in_proj.bias, weights[f"h.{i}.attn.c_attn.bias"], reshape_to=[1, -1])
            copy_tensor(block.self_attn.out_proj.weight, weights[f"h.{i}.attn.c_proj.weight"])
            copy_tensor(block.self_attn.out_proj.bias, weights[f"h.{i}.attn.c_proj.bias"], reshape_to=[1, -1])
            
            # MLP weights
            copy_tensor(block.linear1.weight, weights[f"h.{i}.mlp.c_fc.weight"])
            copy_tensor(block.linear1.bias, weights[f"h.{i}.mlp.c_fc.bias"], reshape_to=[1, -1])
            copy_tensor(block.linear2.weight, weights[f"h.{i}.mlp.c_proj.weight"])
            copy_tensor(block.linear2.bias, weights[f"h.{i}.mlp.c_proj.bias"], reshape_to=[1, -1])
            
            # Norms
            copy_tensor(block.norm1.weight, weights[f"h.{i}.ln_1.weight"])
            copy_tensor(block.norm1.bias, weights[f"h.{i}.ln_1.bias"])
            copy_tensor(block.norm2.weight, weights[f"h.{i}.ln_2.weight"])
            copy_tensor(block.norm2.bias, weights[f"h.{i}.ln_2.bias"])
            
        # Ties weight with wte
        model.lm_head.weight.tensor = model.wte.weight.tensor

    elif model_name == "bert-base-uncased":
        # Map BERT-base
        copy_tensor(model.word_embeddings.weight, weights["bert.embeddings.word_embeddings.weight"])
        copy_tensor(model.position_embeddings.weight, weights["bert.embeddings.position_embeddings.weight"])
        copy_tensor(model.token_type_embeddings.weight, weights["bert.embeddings.token_type_embeddings.weight"])
        copy_tensor(model.ln.weight, weights["bert.embeddings.LayerNorm.weight"])
        copy_tensor(model.ln.bias, weights["bert.embeddings.LayerNorm.bias"])
        
        num_layers = config.get("num_hidden_layers", 12)
        for i in range(num_layers):
            block = model.encoder._modules[str(i)].block
            
            # Concatenate Q, K, V parameters for packed in_proj in MultiHeadAttention
            wq = weights[f"bert.encoder.layer.{i}.attention.self.query.weight"]
            wk = weights[f"bert.encoder.layer.{i}.attention.self.key.weight"]
            wv = weights[f"bert.encoder.layer.{i}.attention.self.value.weight"]
            w_qkv = np.concatenate([wq.T, wk.T, wv.T], axis=1)
            copy_tensor(block.self_attn.in_proj.weight, w_qkv)
            
            bq = weights[f"bert.encoder.layer.{i}.attention.self.query.bias"]
            bk = weights[f"bert.encoder.layer.{i}.attention.self.key.bias"]
            bv = weights[f"bert.encoder.layer.{i}.attention.self.value.bias"]
            b_qkv = np.concatenate([bq, bk, bv])
            copy_tensor(block.self_attn.in_proj.bias, b_qkv, reshape_to=[1, -1])
            
            copy_tensor(block.self_attn.out_proj.weight, weights[f"bert.encoder.layer.{i}.attention.output.dense.weight"].T)
            copy_tensor(block.self_attn.out_proj.bias, weights[f"bert.encoder.layer.{i}.attention.output.dense.bias"], reshape_to=[1, -1])
            
            # MLP
            copy_tensor(block.linear1.weight, weights[f"bert.encoder.layer.{i}.intermediate.dense.weight"].T)
            copy_tensor(block.linear1.bias, weights[f"bert.encoder.layer.{i}.intermediate.dense.bias"], reshape_to=[1, -1])
            copy_tensor(block.linear2.weight, weights[f"bert.encoder.layer.{i}.output.dense.weight"].T)
            copy_tensor(block.linear2.bias, weights[f"bert.encoder.layer.{i}.output.dense.bias"], reshape_to=[1, -1])
            
            # Norms
            copy_tensor(block.norm1.weight, weights[f"bert.encoder.layer.{i}.attention.output.LayerNorm.weight"])
            copy_tensor(block.norm1.bias, weights[f"bert.encoder.layer.{i}.attention.output.LayerNorm.bias"])
            copy_tensor(block.norm2.weight, weights[f"bert.encoder.layer.{i}.output.LayerNorm.weight"])
            copy_tensor(block.norm2.bias, weights[f"bert.encoder.layer.{i}.output.LayerNorm.bias"])
            
        # Pooler
        copy_tensor(model.pooler_dense.weight, weights["bert.pooler.dense.weight"].T)
        copy_tensor(model.pooler_dense.bias, weights["bert.pooler.dense.bias"], reshape_to=[1, -1])

    elif model_name == "resnet18":
        # Map ResNet-18
        copy_tensor(model.conv1.weight, weights["resnet.embedder.embedder.convolution.weight"])
        copy_tensor(model.bn1.weight, weights["resnet.embedder.embedder.normalization.weight"])
        copy_tensor(model.bn1.bias, weights["resnet.embedder.embedder.normalization.bias"])
        copy_tensor(model.bn1.running_mean, weights["resnet.embedder.embedder.normalization.running_mean"])
        copy_tensor(model.bn1.running_var, weights["resnet.embedder.embedder.normalization.running_var"])
        
        # Helper to load BasicBlocks
        def load_block(block, prefix):
            copy_tensor(block.conv1.weight, weights[f"{prefix}.convolution1.weight"])
            copy_tensor(block.bn1.weight, weights[f"{prefix}.normalization1.weight"])
            copy_tensor(block.bn1.bias, weights[f"{prefix}.normalization1.bias"])
            copy_tensor(block.bn1.running_mean, weights[f"{prefix}.normalization1.running_mean"])
            copy_tensor(block.bn1.running_var, weights[f"{prefix}.normalization1.running_var"])
            
            copy_tensor(block.conv2.weight, weights[f"{prefix}.convolution2.weight"])
            copy_tensor(block.bn2.weight, weights[f"{prefix}.normalization2.weight"])
            copy_tensor(block.bn2.bias, weights[f"{prefix}.normalization2.bias"])
            copy_tensor(block.bn2.running_mean, weights[f"{prefix}.normalization2.running_mean"])
            copy_tensor(block.bn2.running_var, weights[f"{prefix}.normalization2.running_var"])
            
            if block.downsample:
                conv_ds = block.downsample._modules["Conv2d_0"]
                bn_ds = block.downsample._modules["BatchNorm2d_1"]
                copy_tensor(conv_ds.weight, weights[f"{prefix}.shortcut.convolution.weight"])
                copy_tensor(bn_ds.weight, weights[f"{prefix}.shortcut.normalization.weight"])
                copy_tensor(bn_ds.bias, weights[f"{prefix}.shortcut.normalization.bias"])
                copy_tensor(bn_ds.running_mean, weights[f"{prefix}.shortcut.normalization.running_mean"])
                copy_tensor(bn_ds.running_var, weights[f"{prefix}.shortcut.normalization.running_var"])

        # Load blocks for layer1 - layer4
        for stage in range(1, 5):
            layer = getattr(model, f"layer{stage}")
            # ResNet has stages indices 0 to 3 corresponding to layer1 to layer4
            stage_idx = stage - 1
            for b in range(2):
                block = layer._modules[f"BasicBlock_{b}"]
                load_block(block, f"resnet.encoder.stages.{stage_idx}.blocks.{b}")
                
        # Final classifier
        copy_tensor(model.fc.weight, weights["classifier.weight"].T)
        copy_tensor(model.fc.bias, weights["classifier.bias"], reshape_to=[1, -1])

    print("Pretrained weights loaded successfully.")
    return model
