"""
Main entry point for the experiments.
"""

import lm_eval
def main():
    """
    Main entry point for the experiments.
    """
    results = lm_eval.simple_evaluate(
        model="hf",
        model_args="pretrained=Qwen/Qwen3-1.7B,device=cuda,dtype=bfloat16",
        tasks=["hellaswag"],
        limit=1000,
    )
    task = "hellaswag"
    metrics = results["results"][task]

    print(metrics)

if __name__ == "__main__":
    main()
