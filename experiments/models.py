"""
Models for the experiments.
"""
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP



class TopkLinear(nn.Linear):
    """
    Linear layer that applies top-k masking to the input or weight.
    """
    def __init__(
        self, 
        *args,
        top_k_ratio: Optional[float] = None,
        prune_input: bool = False,
        **kwargs,
    ):
        """
        Args:
            top_k_ratio: The ratio of the top-k values to keep.
            prune_input: Whether to prune the input.
        """
        print(f"Initializing TopkLinear with top_k_ratio={top_k_ratio} and prune_input={prune_input}")
        super().__init__(*args, **kwargs)
        self.top_k_ratio = top_k_ratio
        self.prune_input = prune_input
        
    
    def prune_weights(self):
        """
        Permanently prune the weights based on top-k ratio.
        Should be called after weights are loaded.
        """
        if self.prune_input or self.top_k_ratio is None:
            return
        
        with torch.no_grad():
            abs_weight = self.weight.abs()
            kth_index = int(abs_weight.size(1) * self.top_k_ratio)
            kth_values, _ = torch.kthvalue(abs_weight, kth_index, dim=1, keepdim=True)
            mask = abs_weight >= kth_values
            # Permanently prune by setting values to zero
            self.weight.data = self.weight.data * mask

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Runs the forward pass.
        """
        if self.top_k_ratio is None:
            return F.linear(input, self.weight, self.bias)
        
        if self.prune_input:
            # Apply absolute top-k masking on input's last dimension (input_dim)
            abs_input = input.abs()
            kth_index = int(abs_input.size(-1) * self.top_k_ratio)
            # Get the k-th largest value along the last dimension
            kth_values, _ = torch.kthvalue(abs_input, kth_index, dim=-1, keepdim=True)
            # Create mask: keep values >= k-th value
            mask = abs_input >= kth_values
            masked_input = input * mask
            return F.linear(masked_input, self.weight, self.bias)
        
        # Weight pruning: weights are already pruned permanently, just use them
        return F.linear(input, self.weight, self.bias)

class TopkQwen3MLP(Qwen3MLP):
    """
    Qwen3 MLP that applies top-k masking to the input or weight.
    """
    def __init__(
        self,
        *args,
        top_k_ratio: Optional[float] = None,
        prune_input: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.top_k_ratio = top_k_ratio
        self.prune_input = prune_input
        
        # Replace the linear layers with TopkLinear versions
        self.gate_proj = TopkLinear(
            in_features=self.gate_proj.in_features,
            out_features=self.gate_proj.out_features,
            bias=self.gate_proj.bias is not None,
            top_k_ratio=top_k_ratio,
            prune_input=prune_input,
        )
        self.up_proj = TopkLinear(
            in_features=self.up_proj.in_features,
            out_features=self.up_proj.out_features,
            bias=self.up_proj.bias is not None,
            top_k_ratio=top_k_ratio,
            prune_input=prune_input,
        )
        self.down_proj = TopkLinear(
            in_features=self.down_proj.in_features,
            out_features=self.down_proj.out_features,
            bias=self.down_proj.bias is not None,
            top_k_ratio=top_k_ratio,
            prune_input=prune_input,
        )
        
    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj