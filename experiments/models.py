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
      kth_values, _ = torch.kthvalue(
          abs_weight, kth_index, dim=1, keepdim=True)
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
      kth_values, _ = torch.kthvalue(
          abs_input, kth_index, dim=-1, keepdim=True)
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
    down_proj = self.down_proj(self.act_fn(
        self.gate_proj(x)) * self.up_proj(x))
    return down_proj



class NMPruneLinear(nn.Linear):
  """
  Linear layer with N:M structured sparsity on weight columns.
  For every M consecutive elements along each column (dim=0), only the
  N elements with the largest absolute values are preserved.
  """

  def __init__(
      self,
      *args,
      n: int = 2,
      m: int = 4,
      prune_input: bool = False,
      **kwargs,
  ):
    super().__init__(*args, **kwargs)
    self.n = n
    self.m = m
    self.prune_input = prune_input

  def prune_weights(self):
    """
    Permanently apply N:M structured pruning along each column.
    Skipped when prune_input=True (pruning happens at forward time instead).
    """
    if self.prune_input:
      return
    with torch.no_grad():
      self.weight.data = self._nm_prune(self.weight.data, self.n, self.m)

  @staticmethod
  def _nm_prune(tensor: torch.Tensor, n: int, m: int) -> torch.Tensor:
    """
    Apply N:M pruning along dim-0 of a 2-D tensor.
    Rows that don't divide evenly by M are left untouched in the
    leftover tail.
    """
    rows, cols = tensor.shape
    full_groups = rows // m
    pruned = tensor.clone()

    if full_groups > 0:
      head = tensor[:full_groups * m, :].reshape(full_groups, m, cols)
      abs_head = head.abs()
      _, topk_idx = abs_head.topk(n, dim=1)
      mask = torch.zeros_like(head, dtype=torch.bool)
      mask.scatter_(1, topk_idx, True)
      pruned[:full_groups * m, :] = (head * mask).reshape(full_groups * m, cols)

    return pruned

  @staticmethod
  def _nm_prune_last_dim(tensor: torch.Tensor, n: int, m: int) -> torch.Tensor:
    """
    Apply N:M pruning along the last dimension of an arbitrary-rank tensor.
    For every M consecutive elements, keep the top-N by absolute value.
    """
    orig_shape = tensor.shape
    L = orig_shape[-1]
    full_groups = L // m
    flat = tensor.reshape(-1, L)
    pruned = flat.clone()

    if full_groups > 0:
      usable = full_groups * m
      head = flat[:, :usable].reshape(flat.size(0), full_groups, m)
      abs_head = head.abs()
      _, topk_idx = abs_head.topk(n, dim=2)
      mask = torch.zeros_like(head, dtype=torch.bool)
      mask.scatter_(2, topk_idx, True)
      pruned[:, :usable] = (head * mask).reshape(flat.size(0), usable)

    return pruned.reshape(orig_shape)

  def forward(self, input: torch.Tensor) -> torch.Tensor:
    if self.prune_input:
      masked_input = self._nm_prune_last_dim(input, self.n, self.m)
      return F.linear(masked_input, self.weight, self.bias)
    return F.linear(input, self.weight, self.bias)


class NMPruneQwen3MLP(Qwen3MLP):
  """
  Qwen3 MLP with N:M structured weight pruning on every linear layer.
  """

  def __init__(
      self,
      *args,
      n: int = 2,
      m: int = 4,
      prune_input: bool = False,
      **kwargs,
  ):
    super().__init__(*args, **kwargs)
    self.n = n
    self.m = m

    self.gate_proj = NMPruneLinear(
        in_features=self.gate_proj.in_features,
        out_features=self.gate_proj.out_features,
        bias=self.gate_proj.bias is not None,
        n=n, m=m,
        prune_input=prune_input,
    )
    self.up_proj = NMPruneLinear(
        in_features=self.up_proj.in_features,
        out_features=self.up_proj.out_features,
        bias=self.up_proj.bias is not None,
        n=n, m=m,
        prune_input=prune_input,
    )
    self.down_proj = NMPruneLinear(
        in_features=self.down_proj.in_features,
        out_features=self.down_proj.out_features,
        bias=self.down_proj.bias is not None,
        n=n, m=m,
        prune_input=prune_input,
    )

  def forward(self, x):
    down_proj = self.down_proj(self.act_fn(
        self.gate_proj(x)) * self.up_proj(x))
    return down_proj

