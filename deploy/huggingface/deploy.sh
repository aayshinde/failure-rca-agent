#!/usr/bin/env bash
# Publish the demo to a Hugging Face Space.
#   1. Create a free account and a Space (SDK: Docker, blank) at https://huggingface.co/new-space
#   2. huggingface-cli login            (or: git credential helper with a write token)
#   3. ./deploy/huggingface/deploy.sh <your-hf-username>/<space-name>
set -euo pipefail
SPACE="${1:?usage: deploy.sh <user>/<space>}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TMP="$(mktemp -d)"
git clone "https://huggingface.co/spaces/$SPACE" "$TMP/space"
cd "$TMP/space"
rsync -a --delete --exclude .git "$ROOT/src" "$ROOT/dashboard" "$ROOT/api.py" "$ROOT/run_pipeline.py" "$ROOT/requirements.txt" ./
cp "$ROOT/deploy/huggingface/Dockerfile" Dockerfile
cp "$ROOT/deploy/huggingface/SPACE_README.md" README.md
git add -A && git commit -m "Deploy demo" && git push
echo "Live at https://huggingface.co/spaces/$SPACE (first build takes ~5-8 minutes)"
