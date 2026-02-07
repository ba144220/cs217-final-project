"""
Main entry point for the experiments.
"""
import torch
import torch.nn as nn
from transformers import Qwen3ForCausalLM, AutoTokenizer
import lm_eval
from lm_eval.models.huggingface import HFLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP

from models import TopkQwen3MLP

MODEL_NAME = "Qwen/Qwen3-1.7B"
TOP_K_RATIO = 0.5
PRUNE_INPUT = False


def replace_mlp_layers(model, top_k_ratio, prune_input):
    """
    Replace all Qwen3MLP layers with TopkQwen3MLP layers.
    """
    for name, module in model.named_modules():
        if isinstance(module, Qwen3MLP) and not isinstance(module, TopkQwen3MLP):
            # Get the parent module and the attribute name
            parent_name = '.'.join(name.split('.')[:-1])
            attr_name = name.split('.')[-1]
            
            if parent_name:
                parent = model.get_submodule(parent_name)
            else:
                parent = model
            
            # Create new TopkQwen3MLP with same config as original
            new_mlp = TopkQwen3MLP(
                config=module.config,
                top_k_ratio=top_k_ratio,
                prune_input=prune_input,
            )
            
            # Copy weights from original to new module
            new_mlp.gate_proj.weight.data = module.gate_proj.weight.data.clone()
            new_mlp.up_proj.weight.data = module.up_proj.weight.data.clone()
            new_mlp.down_proj.weight.data = module.down_proj.weight.data.clone()
            
            if module.gate_proj.bias is not None:
                new_mlp.gate_proj.bias.data = module.gate_proj.bias.data.clone()
            if module.up_proj.bias is not None:
                new_mlp.up_proj.bias.data = module.up_proj.bias.data.clone()
            if module.down_proj.bias is not None:
                new_mlp.down_proj.bias.data = module.down_proj.bias.data.clone()
            
            # Prune weights permanently after loading (for weight pruning)
            new_mlp.gate_proj.prune_weights()
            new_mlp.up_proj.prune_weights()
            new_mlp.down_proj.prune_weights()
            
            # Replace the module
            setattr(parent, attr_name, new_mlp)
            print(f"Replaced {name} with TopkQwen3MLP")
    
    return model


def main():
    """
    Main entry point for the experiments.
    """

    model = Qwen3ForCausalLM.from_pretrained(
        MODEL_NAME, 
        device_map="auto", 
        dtype=torch.bfloat16,
        tie_word_embeddings=False
    )
    
    # Replace all Qwen3MLP layers with TopkQwen3MLP
    model = replace_mlp_layers(model, TOP_K_RATIO, PRUNE_INPUT)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        device="cuda",
        dtype=torch.bfloat16,
    )

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=["hellaswag"],
        limit=1000,
    )
    task = "hellaswag"
    metrics = results["results"][task]

    print(metrics)

if __name__ == "__main__":
    main()
