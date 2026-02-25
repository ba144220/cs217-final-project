"""
Utils for the experiments.
"""

from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP
from models import TopkQwen3MLP, NMPruneQwen3MLP

def replace_mlp_layers_topk(model, top_k_ratio, prune_input):
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


def replace_mlp_layers_nm(model, n, m, prune_input=False):
  """
  Replace all Qwen3MLP layers with NMPruneQwen3MLP layers.
  """
  for name, module in model.named_modules():
    if isinstance(module, Qwen3MLP) and not isinstance(module, NMPruneQwen3MLP):
      parent_name = '.'.join(name.split('.')[:-1])
      attr_name = name.split('.')[-1]

      if parent_name:
        parent = model.get_submodule(parent_name)
      else:
        parent = model

      new_mlp = NMPruneQwen3MLP(
          config=module.config,
          n=n, m=m,
          prune_input=prune_input,
      )

      new_mlp.gate_proj.weight.data = module.gate_proj.weight.data.clone()
      new_mlp.up_proj.weight.data = module.up_proj.weight.data.clone()
      new_mlp.down_proj.weight.data = module.down_proj.weight.data.clone()

      if module.gate_proj.bias is not None:
        new_mlp.gate_proj.bias.data = module.gate_proj.bias.data.clone()
      if module.up_proj.bias is not None:
        new_mlp.up_proj.bias.data = module.up_proj.bias.data.clone()
      if module.down_proj.bias is not None:
        new_mlp.down_proj.bias.data = module.down_proj.bias.data.clone()

      new_mlp.gate_proj.prune_weights()
      new_mlp.up_proj.prune_weights()
      new_mlp.down_proj.prune_weights()

      setattr(parent, attr_name, new_mlp)
      print(f"Replaced {name} with NMPruneQwen3MLP (n={n}, m={m})")

  return model

