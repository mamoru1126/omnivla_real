#!/bin/bash
# 公式の重みを /checkpoints に取得する (Hugging Face, git lfs).
#   bash scripts/download_checkpoints.sh            # edge と 7B の両方
#   bash scripts/download_checkpoints.sh edge       # OmniVLA-edge だけ (約 1GB 未満)
#   bash scripts/download_checkpoints.sh 7b         # OmniVLA 7B だけ (約 16GB)
set -euo pipefail
DEST=${CHECKPOINT_DIR_IN_CONTAINER:-/checkpoints}
WHAT=${1:-all}
git lfs install --skip-repo >/dev/null 2>&1 || true
get() {
  local name=$1
  if [ -d "$DEST/$name/.git" ]; then
    echo "== $name: already exists, pulling"; (cd "$DEST/$name" && git pull -q && git lfs pull)
  else
    echo "== $name: cloning"; git clone "https://huggingface.co/NHirose/$name" "$DEST/$name"
  fi
}
mkdir -p "$DEST"
case "$WHAT" in
  edge) get omnivla-edge ;;
  7b) get omnivla-original ;;
  all) get omnivla-edge; get omnivla-original ;;
  *) echo "usage: $0 [all|edge|7b]"; exit 1 ;;
esac
ls -la "$DEST"
