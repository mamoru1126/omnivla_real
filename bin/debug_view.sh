#!/usr/bin/env bash
# デバッグ画面 (debug_view.py) を nav コンテナの中で動かす (ROS 2 Humble). Orin の上で、bin/jetson.sh up のあとに.
#   bin/debug_view.sh                # → ブラウザで http://192.168.0.200:8090 (Ctrl-C で終わり、まとめを表示)
#   bin/debug_view.sh --port 8091    # オプションは debug_view.py にそのまま渡す
# debug_view.py は tools/ (リポジトリ) か log/ (ネットにつながらない Orin に置いた場合) のもの.
# このファイルも log/ に置いて bash log/debug_view.sh で動かせる.
set -euo pipefail
case "${1:-}" in -h | --help) sed -n '2,/^[^#]/{/^#/{s/^# \{0,1\}//;p}}' "$0"; exit 0 ;; esac
ROOT=$(cd "$(dirname "$0")/.." && pwd)          # リポジトリ (このファイルは bin/ か log/ にある)
cd "$ROOT"
PY=tools/debug_view.py
[ -f "$PY" ] || PY=log/debug_view.py
[ -f "$PY" ] || { echo "debug_view.py が見つかりません ($ROOT/tools/ か $ROOT/log/ に置いてください)"; exit 1; }
DC=(docker compose -f docker-compose.jetson.yml -f docker-compose.jetson-humble.yml)
if [ -z "$("${DC[@]}" ps -q nav 2> /dev/null)" ]; then
    echo "nav コンテナが動いていません. 先に bin/jetson.sh up"
    exit 1
fi
TTY=()
if ! { [ -t 0 ] && [ -t 1 ]; }; then TTY=(-T); fi
echo "ブラウザ: http://$(hostname -I | awk '{print $1}'):8090  (ATMOBI PC からは http://192.168.0.200:8090)"
exec "${DC[@]}" exec "${TTY[@]}" nav bash -c "source /opt/omnivla_ws/install/setup.bash && python3 /workspace/$PY $*"
