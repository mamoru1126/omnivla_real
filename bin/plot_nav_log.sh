#!/usr/bin/env bash
# 走行ログ (ナビゲーションノードの記録) を図とレポートにする. 図はログのフォルダに出る.
#
#   bin/plot_nav_log.sh                         # 学習 PC で最後に流した bag (runs/replay/nav/latest)
#   bin/plot_nav_log.sh log/nav/latest          # Jetson から持ってきた走行ログ (リポジトリの中のパスでもよい)
#   bin/plot_nav_log.sh runs/replay/nav/20261008_120000 --every 5
source "$(dirname "$0")/_lib.sh"
DIR=/runs/replay/nav/latest
if [ $# -gt 0 ] && [[ $1 != -* ]]; then
    DIR=$(cpath runs "" "$1")
    shift
fi
case $DIR in */latest) need "${DIR%/latest}" ;; *) need "$DIR" ;; esac   # latest は一番新しい走行のこと
in_shell python3 tools/plot_nav_log.py "$DIR" "$@"
say "図: $(hpath "$DIR")"
