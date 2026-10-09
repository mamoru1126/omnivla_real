#!/usr/bin/env bash
# 学習 PC の準備: .env を作り、イメージ (学習用・ROS 1) をビルドして、公式の重みを取ってくる.
#
#   bin/setup.sh          # 重みは 7B と edge の両方
#   bin/setup.sh edge     # edge の重みだけ
#   bin/setup.sh 7b       # 7B の重みだけ
#   bin/setup.sh none     # 重みは取らない (ビルドだけ)
source "$(dirname "$0")/_lib.sh"
WHAT=${1:-all}
case $WHAT in all | edge | 7b | none) ;; *) usage 1 ;; esac
ensure_env
# docker に root で作られないよう、先に作っておく
for k in bags data runs checkpoints; do mkdir -p "$(host_dir "$k")"; done
say "イメージをビルドします (初回は 30 分ほどかかります)"
pc build
say "単体テスト (1 分ほど)"
if ! in_shell python3 -m pytest -q tests; then
    say "注意: 単体テストに失敗しました (イメージはできています). 上のログを確認してください"
fi
if [ "$WHAT" != none ]; then
    in_shell bash scripts/download_checkpoints.sh "$WHAT"
fi
say "準備できました. 次は bag を $(host_dir bags) に置いて: bin/bag_info.sh <bag>"
