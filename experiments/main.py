"""
Main entry point for the experiments.
"""

import lm_eval
from lm_eval.models.huggingface import HFLM
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-1.7B"

def main():
    """
    Main entry point for the experiments.
    """
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        device_map="auto", 
        dtype=torch.bfloat16,
        tie_word_embeddings=False
    )
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
