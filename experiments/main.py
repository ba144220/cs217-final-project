"""
Main entry point for the experiments.
"""

import lm_eval
import json
def main():
    """
    Main entry point for the experiments.
    """
    results = lm_eval.simple_evaluate(
        model="hf",
        model_args="pretrained=arnir0/Tiny-LLM,device=cpu",
        tasks=["hellaswag"],
        limit=100,
    )
    task = "hellaswag"
    metrics = results["results"][task]

    print(metrics)

    
if __name__ == "__main__":
    main()
