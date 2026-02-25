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
  match = re.match(r"^(\d+)-(\d+)-(act|weight)$", name)
  if not match:
    return None
  return int(match.group(1)), int(match.group(2)), match.group(3)


def main():
  rows = []
  for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json"))):
    parsed = parse_filename(path)
    if parsed is None:
      continue
    n, m, target = parsed

    with open(path) as f:
      data = json.load(f)

    accs = {}
    for task in TASKS:
      if task in data:
        accs[task] = data[task].get("acc,none", None)
      else:
        accs[task] = None

    rows.append((n, m, target, accs))

  rows.sort(key=lambda r: (r[0] / r[1], r[1], r[2]))

  header = ",".join(["N", "M", "Target"] + TASKS)
  print(header)

  for n, m, target, accs in rows:
    cols = [str(n), str(m), target]
    for task in TASKS:
      v = accs[task]
      cols.append(f"{v:.4f}" if v is not None else "N/A")
    print(",".join(cols))


if __name__ == "__main__":
  main()
