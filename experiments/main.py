"""
Main entry point for the experiments.
"""
import torch
import torch.nn as nn
from transformers import Qwen3ForCausalLM, AutoTokenizer
import lm_eval
from lm_eval.models.huggingface import HFLM
from utils import replace_mlp_layers_nm

MODEL_NAME = "Qwen/Qwen3-1.7B"
TOP_K_RATIO = 0.5
PRUNE_INPUT = False


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
  # model = replace_mlp_layers_topk(model, TOP_K_RATIO, PRUNE_INPUT)
  model = replace_mlp_layers_nm(model, 1, 16, PRUNE_INPUT)

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
