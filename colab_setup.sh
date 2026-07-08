#!/bin/bash
# One-shot setup for a fresh Colab session, run from this repo locally:
#   ./colab_setup.sh <session-name>
#
# Creates nothing itself — assumes `colab new -s <name> --gpu A100` already ran.
# Steps: install locked deps, remove Colab's preinstalled torchvision/torchaudio
# builds that clash with our locked torch (torchvision is reinstalled from the
# lock; torchaudio we don't use), restart the kernel, set the HF token, and
# upload the data files.
set -euo pipefail

SESSION="${1:-ds6051}"

echo "== installing locked requirements =="
colab install -s "$SESSION" -r requirements.txt

echo "== removing stale torch extras (torchaudio; torchvision comes from our lock) =="
echo 'import subprocess; r = subprocess.run(["uv", "pip", "uninstall", "--system", "torchaudio"], capture_output=True, text=True); print(r.stderr.strip()[-200:])' | colab exec -s "$SESSION"

echo "== restarting kernel =="
colab restart-kernel -s "$SESSION"

echo "== setting HF token and verifying imports =="
TOKEN=$(uv run hf auth token)
printf 'import os\nos.environ["HF_TOKEN"] = "%s"\nos.makedirs("/content/data", exist_ok=True)\nimport transformers, torch\nprint("ready | transformers", transformers.__version__, "| torch", torch.__version__, "| cuda:", torch.cuda.is_available())\n' "$TOKEN" | colab exec -s "$SESSION"

echo "== uploading data files =="
[ -f data/eval_prompts.jsonl ] && colab upload -s "$SESSION" data/eval_prompts.jsonl /content/data/eval_prompts.jsonl
[ -f data/responses.json ] && colab upload -s "$SESSION" data/responses.json /content/data/responses.json
[ -f judge_prompts.md ] && colab upload -s "$SESSION" judge_prompts.md /content/judge_prompts.md

echo "== done =="
