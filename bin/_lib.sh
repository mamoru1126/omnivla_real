# shellcheck shell=bash
# bin/*.sh の共通部分 (各スクリプトが source する. 単体では使わない).
#
# - リポジトリの場所に移動してから docker compose を呼ぶので、どこから実行してもよい。
# - bag や topomap は「名前」「ホストのパス」「コンテナの中のパス」のどれで指定してもよい。
#     run1.bag / ./bags/run1.bag / /bags/run1.bag  →  /bags/run1.bag
#     course_a / ./data/topomaps/course_a          →  /data/topomaps/course_a
# - 値は 環境変数 → .env → 既定値 の順に見る。
# - OMNIVLA_DRY_RUN=1 にすると docker を実行せず、実行するコマンドを表示するだけ (確認・テスト用)。
set -euo pipefail
CALLER_PWD=$PWD
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
ME=$(basename "$0" .sh)

say() { echo "[$ME] $*"; }
die() { echo "[$ME] $*" >&2; exit 1; }

# スクリプト先頭のコメントを使い方として表示する (usage 1 で終了コード 1)
# shellcheck disable=SC2120
usage() {
    sed -n '2,/^[^#]/{/^#/{s/^# \{0,1\}//;p}}' "$0"
    exit "${1:-0}"
}
case "${1:-}" in -h | --help | help) usage ;; esac

# 環境変数 → .env → 既定値
env_get() {
    local k=$1 d=${2:-} v=""
    if [ -n "${!k:-}" ]; then
        printf '%s\n' "${!k}"
        return
    fi
    if [ -f "$ROOT/.env" ]; then
        v=$(sed -n -E "s/^[[:space:]]*(export[[:space:]]+)?$k=(.*)$/\2/p" "$ROOT/.env" | tail -1 | sed -E 's/[[:space:]]+#.*$//')
        v=${v%$'\r'}
        case $v in \"*\") v=${v:1:${#v}-2} ;; \'*\') v=${v:1:${#v}-2} ;; esac
    fi
    printf '%s\n' "${v:-$d}"
}

ensure_env() {
    local src=.env.example
    if on_jetson && [ -f "$ROOT/.env.jetson.example" ]; then src=.env.jetson.example; fi
    if [ ! -f "$ROOT/.env" ] && [ "${OMNIVLA_DRY_RUN:-0}" != 1 ]; then
        cp "$ROOT/$src" "$ROOT/.env"
        say ".env がなかったので $src から作りました (重み・topomap などはここに書きます)"
    fi
}

# コンテナにマウントされるホストのディレクトリ (絶対パス). bags | data | runs | checkpoints
host_dir() {
    local var
    case $1 in
        bags) var=BAG_DIR ;;
        data) var=DATA_DIR ;;
        runs) var=RUNS_DIR ;;
        checkpoints) var=CHECKPOINT_DIR ;;
        *) die "host_dir: $1" ;;
    esac
    (cd "$ROOT" && realpath -m -- "$(env_get "$var" "./$1")")
}

# 名前 / ホストのパス / コンテナのパス → コンテナの中のパス
#   cpath KIND SUBDIR ARG    例: cpath data topomaps course_a → /data/topomaps/course_a
cpath() {
    local kind=$1 sub=$2 p=$3 h k d
    if [ -z "$p" ]; then
        echo ""
        return
    fi
    case $p in
        /bags | /bags/* | /data | /data/* | /runs | /runs/* | /checkpoints | /checkpoints/* | /workspace | /workspace/*)
            echo "$p"
            return
            ;;
    esac
    h=$(cd "$CALLER_PWD" && realpath -m -- "$p")
    if [ -e "$h" ]; then
        for k in bags data runs checkpoints; do
            d=$(host_dir "$k")
            if [ "$h" = "$d" ] || [[ $h == "$d"/* ]]; then
                echo "/$k${h#"$d"}"
                return
            fi
        done
        if [[ $h == "$ROOT"/* ]]; then
            echo "/workspace${h#"$ROOT"}"
            return
        fi
        die "$p はコンテナから見えません. $(host_dir "$kind") の下に置いてください"
    fi
    echo "/$kind${sub:+/$sub}/$p"
}

# これから作るファイル (出力先) のパス → コンテナの中のパス. マウントされたディレクトリの下でないとだめ
cpath_new() {
    local p=$1 h k d
    case $p in
        /bags/* | /data/* | /runs/* | /checkpoints/* | /workspace/*)
            echo "$p"
            return
            ;;
    esac
    h=$(cd "$CALLER_PWD" && realpath -m -- "$p")
    for k in bags data runs checkpoints; do
        d=$(host_dir "$k")
        if [[ $h == "$d"/* ]]; then
            echo "/$k${h#"$d"}"
            return
        fi
    done
    if [[ $h == "$ROOT"/* ]]; then
        echo "/workspace${h#"$ROOT"}"
        return
    fi
    die "$p はコンテナから見えない場所です. $(host_dir runs) の下などを指定してください"
}

# コンテナの中のパス → ホストのパス
hpath() {
    local p=$1 k
    for k in bags data runs checkpoints; do
        case $p in /"$k" | /"$k"/*)
            echo "$(host_dir "$k")${p#/"$k"}"
            return
            ;;
        esac
    done
    case $p in /workspace | /workspace/*)
        echo "$ROOT${p#/workspace}"
        return
        ;;
    esac
    echo "$p"
}

# コンテナのパスに対応するものがホストにあるか
need() {
    [ -e "$(hpath "$1")" ] || die "見つかりません: $1 (ホストでは $(hpath "$1"))"
}

# bag (カンマ区切りの分割 bag も可) → コンテナのパス (カンマ区切り)
cbags() {
    local items out=() b c
    IFS=',' read -r -a items <<< "$1"
    for b in "${items[@]}"; do
        c=$(cpath bags "" "$b")
        need "$c"
        out+=("$c")
    done
    local IFS=','
    echo "${out[*]}"
}

# bag の名前 (出力先の名前に使う): /bags/run1_0.bag,/bags/run1_1.bag → run1_0
bag_name() {
    local b=${1%%,*}
    b=$(basename "${b%/}")
    echo "${b%.bag}"
}

topomap_path() {
    local t
    t=$(cpath data topomaps "$1")
    need "$t"
    echo "$t"
}

# --- docker ---
docker() {
    if [ "${OMNIVLA_DRY_RUN:-0}" = 1 ]; then
        echo "+ docker $*"
    else
        command docker "$@"
    fi
}
# イメージがなければ作る: ensure_image IMAGE ビルドするコマンド...
ensure_image() {
    local img=$1
    shift
    if [ "${OMNIVLA_DRY_RUN:-0}" != 1 ] && ! command docker image inspect "$img" > /dev/null 2>&1; then
        say "$img がないのでビルドします"
        "$@"
    fi
}
TTY=()
if ! { [ -t 0 ] && [ -t 1 ]; }; then TTY=(-T); fi

# Jetson の上か (OMNIVLA_JETSON=1 / 0 で決め打ちもできる). Jetson では学習用イメージの代わりに推論サーバのイメージを使う
on_jetson() {
    case ${OMNIVLA_JETSON:-} in
        1 | true) return 0 ;;
        0 | false) return 1 ;;
    esac
    [ -f /etc/nv_tegra_release ]
}
if on_jetson; then export REPLAY_POLICY_COMPOSE=docker-compose.jetson.yml; fi  # docker-compose.replay.yml の推論サーバ
pc() { docker compose "$@"; }                                 # 学習 PC (docker-compose.yml)
jetson() { docker compose -f docker-compose.jetson.yml "$@"; }  # Jetson
replay() { docker compose -f docker-compose.replay.yml "$@"; }  # bag をロボットの代わりにして実機と同じ構成
# 1 コマンド動かす: 学習 PC は学習用コンテナ、Jetson は推論サーバのイメージ (torch + bag の読み込み)
in_shell() {
    if on_jetson; then
        ensure_image omnivla_real:jetson-policy jetson build policy
        jetson run --rm --no-deps "${TTY[@]}" policy "$@"
    else
        pc run --rm "${TTY[@]}" shell "$@"
    fi
}
# Jetson で bag を流す前の注意 (nav・推論サーバのコンテナは実機の走行と同じものを使う)
jetson_replay_note() {
    on_jetson || return 0
    say "Jetson で bag を流します. 実機の走行用の nav は bag 用に置き換わります (終わったら bin/replay.sh down → bin/jetson.sh up)"
    say "ROS master はこの Jetson の中だけ. 念のためロボットのモーターは切っておいてください"
}
# 学習 PC でしかできないもの
not_on_jetson() {
    if on_jetson; then die "$* は学習 PC で実行してください (Jetson では机上評価・bag での確認・動画・図だけ)"; fi
}
