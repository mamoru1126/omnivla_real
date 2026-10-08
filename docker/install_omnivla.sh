#!/bin/bash
# OmniVLA (コミット固定) を /opt/OmniVLA に置き、prismatic を「依存を最小限にして」使えるようにする.
#  - prismatic/__init__.py などを空にする: そのままだと import prismatic で RLDS 学習用のコード
#    (tensorflow, dlimp, draccus, wandb ...) まで読み込まれ、Jetson では入れられない依存が必要になるため。
#    本リポジトリが使うのは prismatic.extern.hf / models.action_heads / models.projectors /
#    training.train_utils / vla.action_tokenizer / vla.constants / models.backbones.llm.prompting だけ。
#  - pip install はせず PYTHONPATH に /opt/OmniVLA を足す
set -euo pipefail
REPO=${OMNIVLA_REPO:-https://github.com/NHirose/OmniVLA.git}
COMMIT=${OMNIVLA_COMMIT:-5182600cb4a9ee07684e17cdd2a6cbafc56b8a68}
ROOT=${OMNIVLA_ROOT:-/opt/OmniVLA}
git init -q "$ROOT"
cd "$ROOT"
git remote add origin "$REPO"
git fetch -q --depth 1 origin "$COMMIT"
git checkout -q FETCH_HEAD
for f in prismatic/__init__.py prismatic/models/__init__.py prismatic/training/__init__.py prismatic/vla/__init__.py; do
  echo "# emptied by omnivla_real/docker/install_omnivla.sh (avoid importing RLDS / tensorflow code)" > "$f"
done
echo "OmniVLA $COMMIT -> $ROOT (slim prismatic)"
