"""
Extract a summary table from experiments/results/*.json files.
"""
import json
import glob
import os
import re

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
TASKS = ["arc_challenge", "hellaswag", "mmlu", "winogrande"]


def parse_filename(path):
  name = os.path.splitext(os.path.basename(path))[0]

  nm_match = re.match(r"^(\d+)-(\d+)-(act|weight)$", name)
  if nm_match:
    return "nm", int(nm_match.group(1)), int(nm_match.group(2)), nm_match.group(3)

  topk_match = re.match(r"^topk-(\d+)-(act|weight)$", name)
  if topk_match:
    return "topk", int(topk_match.group(1)), None, topk_match.group(2)

  if name == "baseline":
    return "baseline", None, None, "none"

  return None, None, None, None


def load_accs(path):
  with open(path) as f:
    data = json.load(f)
  return {task: data[task].get("acc,none", None) if task in data else None
          for task in TASKS}


def main():
  nm_rows = []
  topk_rows = []
  baseline_row = None

  for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json"))):
    kind, a, b, target = parse_filename(path)
    accs = load_accs(path)

    if kind == "nm":
      nm_rows.append((a, b, target, accs))
    elif kind == "topk":
      topk_rows.append((a, target, accs))
    elif kind == "baseline":
      baseline_row = accs

  # N:M table
  nm_rows.sort(key=lambda r: (r[0] / r[1], r[1], r[2]))
  header = ",".join(["N", "M", "Target"] + TASKS)
  print("=== N:M Pruning ===")
  print(header)
  for n, m, target, accs in nm_rows:
    cols = [str(n), str(m), target]
    for task in TASKS:
      v = accs[task]
      cols.append(f"{v:.4f}" if v is not None else "N/A")
    print(",".join(cols))

  # Top-k table
  topk_rows.sort(key=lambda r: (r[0], r[1]))
  print()
  print("=== Top-k Pruning ===")
  header = ",".join(["K", "Target"] + TASKS)
  print(header)
  for k, target, accs in topk_rows:
    cols = [str(k), target]
    for task in TASKS:
      v = accs[task]
      cols.append(f"{v:.4f}" if v is not None else "N/A")
    print(",".join(cols))

  # Baseline
  if baseline_row:
    print()
    print("=== Baseline (no pruning) ===")
    header = ",".join(TASKS)
    print(header)
    cols = []
    for task in TASKS:
      v = baseline_row[task]
      cols.append(f"{v:.4f}" if v is not None else "N/A")
    print(",".join(cols))


if __name__ == "__main__":
  main()
