"""
Main entry point for the experiments.
"""
import argparse
import json
import os

import torch
from transformers import Qwen3ForCausalLM, AutoTokenizer
import lm_eval
from lm_eval.models.huggingface import HFLM
from utils import replace_mlp_layers_nm, replace_mlp_layers_topk

MODEL_NAME = "Qwen/Qwen3-1.7B"


def parse_args():
  parser = argparse.ArgumentParser(description="N:M pruning experiments")
  parser.add_argument("--n", type=int, default=None,
                      help="Number of values to keep per group")
  parser.add_argument("--m", type=int, default=None,
                      help="Group size")
  parser.add_argument("--k", type=int, default=None,
                      help="Denominator of top-k ratio")
  parser.add_argument("--prune-input", action="store_true",
                      help="Prune input activations instead of weights")
  return parser.parse_args()


def main():
  args = parse_args()

  model = Qwen3ForCausalLM.from_pretrained(
      MODEL_NAME,
      device_map="auto",
      dtype=torch.bfloat16,
      tie_word_embeddings=False
  )

  if args.n is not None and args.m is not None:
    assert args.k is None, "Cannot specify both n and m and k"
    assert args.n != args.m, "n and m must be different"
    assert args.n < args.m, "n must be less than m"
    model = replace_mlp_layers_nm(model, args.n, args.m, args.prune_input)
  elif args.k is not None:
    model = replace_mlp_layers_topk(model, 1 - 1/args.k, args.prune_input)

  else:
    print("No pruning specified, running baseline")

  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
  lm = HFLM(
      pretrained=model,
      tokenizer=tokenizer,
      device="cuda",
      dtype=torch.bfloat16,
  )

  tasks = [
      "hellaswag",       # commonsense reasoning (sentence completion)
      "winogrande",      # commonsense reasoning (pronoun resolution)
      "arc_challenge",   # science QA (hard)
      "mmlu",            # broad knowledge across 57 subjects
  ]

  results = lm_eval.simple_evaluate(
      model=lm,
      tasks=tasks,
      # limit=100,
  )

  prune_target = "act" if args.prune_input else "weight"
  os.makedirs("results", exist_ok=True)

  if args.n is not None and args.m is not None:
    out_path = f"results/{args.n}-{args.m}-{prune_target}.json"
  elif args.k is not None:
    out_path = f"results/topk-{args.k}-{prune_target}.json"
  else:
    out_path = f"results/baseline.json"

  with open(out_path, "w") as f:
    json.dump(results["results"], f, indent=2)
  print(f"Results saved to {out_path}")
  print(results["results"])


if __name__ == "__main__":
  main()
