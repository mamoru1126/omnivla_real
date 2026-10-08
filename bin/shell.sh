#!/usr/bin/env bash
# 学習用コンテナに入る (GPU, /bags /data /runs /checkpoints がマウントされた状態).
#
#   bin/shell.sh                          # bash に入る
#   bin/shell.sh python3 tools/xxx.py ... # コマンドを 1 つ実行して終わる
source "$(dirname "$0")/_lib.sh"
ensure_env
if [ $# -eq 0 ]; then
    in_shell bash
else
    in_shell "$@"
fi
