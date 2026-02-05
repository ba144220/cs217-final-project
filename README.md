# CS217 Final Project

# ML Experiments Setup

## Setup

```bash
git submodule update --init --recursive
uv sync
```

## Submodules

This project uses the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) as a submodule in `third_party/lm-evaluation-harness`. It will be automatically installed when you run `uv sync`.

To update the submodule to the latest version:

```bash
git submodule update --remote
uv sync
```
