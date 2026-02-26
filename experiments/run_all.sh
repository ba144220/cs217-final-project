#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Baseline (n == m, no pruning)
# echo "=== Running baseline (1:1, no pruning) ==="
# python main.py --n 1 --m 1

# (n, m) configs to test
# configs="2,4 4,8 2,8"
# configs="2,16 4,16 8,16"
# configs="4,32 8,32 16,32"

# for cfg in $configs; do
#   n="${cfg%,*}"
#   m="${cfg#*,}"

#   echo "=== Running n=$n m=$m weight pruning ==="
#   python main.py --n "$n" --m "$m"

#   echo "=== Running n=$n m=$m input pruning ==="
#   python main.py --n "$n" --m "$m" --prune-input
# done

# Top-k experiments
for k in 2 4 8; do
  echo "=== Running topk k=$k weight pruning ==="
  python main.py --k "$k"

  echo "=== Running topk k=$k input pruning ==="
  python main.py --k "$k" --prune-input
done

echo "=== All runs complete ==="
echo "Results:"
ls -1 results/*.json
