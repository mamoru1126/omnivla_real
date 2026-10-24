#!/usr/bin/env bash
# 作業 PC から Orin を扱う (Orin はネットにつながらない. LAN の ssh だけで動く). 作業 PC で実行する.
#
#   bin/orin.sh sync                 # この PC のリポジトリ (今のブランチ) を Orin に送って、Orin 側を更新 (git bundle)
#   bin/orin.sh field <サブコマンド>  # Orin で bin/field.sh を実行 (例: bin/orin.sh field check / field start)
#   bin/orin.sh fetch [--bags] [--images]   # Orin の記録を debug_log/orin_<日時>/ に持ってくる (既定は bag 本体と画像を除く)
#                                    #   rsync が無ければ ssh + tar で (OMNIVLA_NO_RSYNC=1 で常にそうする). sync は scp だけ
#   bin/orin.sh ssh                  # Orin に入る (リポジトリのフォルダで)
#   bin/orin.sh bundle [ファイル]     # USB で運ぶ用の更新ファイル (Orin で: git fetch <ファイル> jetson && git merge --ff-only FETCH_HEAD,
#                                    #   2 回目からは bin/field.sh update <ファイル>)
#
# Orin の指定 (環境変数か .env):
#   ORIN_HOST  ~/.ssh/config の名前か user@IP (既定 orin)
#   ORIN_PORT  ssh のポート (22 以外のとき. 例 4003. ~/.ssh/config に書いてあれば不要)
#   ORIN_REPO  Orin のリポジトリ (既定 omnivla_real = ホームの下)
# 重み・topomap・bag を送るのは bin/to_jetson.sh <ホスト> (ポートは ~/.ssh/config に書く)
source "$(dirname "$0")/_lib.sh"

HOST=$(env_get ORIN_HOST orin)
PORT=$(env_get ORIN_PORT "")
REPO=$(env_get ORIN_REPO omnivla_real)
SSH=(ssh)
SCP=(scp)
RSH=ssh
if [ -n "$PORT" ]; then
    SSH+=(-p "$PORT")
    SCP+=(-P "$PORT")
    RSH="ssh -p $PORT"
fi
if [ "${OMNIVLA_DRY_RUN:-0}" = 1 ]; then
    run() { echo "+ $*"; }
else
    run() { "$@"; }
fi

CMD=${1:-}
shift || true
case $CMD in
    sync | bundle)
        BR=$(git rev-parse --abbrev-ref HEAD)
        git diff --quiet HEAD -- 2> /dev/null || say "注意: まだコミットしていない変更は送られません (送るのはコミット済みの $BR)"
        if [ "$CMD" = bundle ]; then
            F=${1:-$CALLER_PWD/omnivla_$BR.bundle}
            run git bundle create "$F" "$BR"
            say "-> $F  (USB などで Orin へ. Orin のリポジトリで: git fetch $(basename "$F") $BR && git merge --ff-only FETCH_HEAD)"
            exit 0
        fi
        TMP=$(mktemp -d)
        trap 'rm -rf "$TMP"' EXIT
        B=$TMP/omnivla_$BR.bundle
        run git bundle create "$B" "$BR"
        say "Orin ($HOST) に送ります: $BR $(git log --oneline -1)"
        run "${SCP[@]}" "$B" "$HOST:/tmp/omnivla_$BR.bundle"
        # Orin 側: 同じブランチなら早送りで取り込む (Orin でしたコミットや、同じファイルの変更があれば止まる)
        REMOTE="set -e; cd $REPO
cur=\$(git rev-parse --abbrev-ref HEAD)
if [ \"\$cur\" != \"$BR\" ]; then echo \"[orin] Orin のブランチは \$cur です (送ったのは $BR). Orin で git checkout $BR してからもう一度\"; exit 1; fi
git fetch -q /tmp/omnivla_$BR.bundle $BR
if git merge --ff-only FETCH_HEAD; then git log --oneline -1; else echo '[orin] 取り込めませんでした. Orin で git status を確認 (Orin で変えたファイルとぶつかっている)'; exit 1; fi"
        run "${SSH[@]}" "$HOST" "$REMOTE"
        say "更新しました. 動いているノードに反映するには: bin/orin.sh field up --topomap <名前> (イメージの作り直しは不要)"
        ;;
    field)
        Q=$(printf ' %q' "$@")
        run "${SSH[@]}" -t "$HOST" "cd $REPO && bin/field.sh$Q"
        ;;
    fetch)
        BAGS=0
        IMAGES=0
        while [ $# -gt 0 ]; do
            case $1 in
                --bags) BAGS=1 ;;
                --images) IMAGES=1 ;;
                *) die "知らないオプション: $1" ;;
            esac
            shift
        done
        EXCL=()
        TEXCL=()
        [ "$BAGS" = 1 ] || {
            EXCL+=(--exclude '*.db3' --exclude '*.mcap')
            TEXCL+=("--exclude=*.db3" "--exclude=*.mcap")
        }
        [ "$IMAGES" = 1 ] || {
            EXCL+=(--exclude 'frames/' --exclude 'raw/' --exclude 'goals/' --exclude 'debug/')
            TEXCL+=(--exclude=frames --exclude=raw --exclude=goals --exclude=debug)
        }
        # rsync が両方に無ければ ssh + tar で持ってくる (同じものを除く)
        USE_RSYNC=1
        if [ "${OMNIVLA_NO_RSYNC:-0}" = 1 ]; then
            USE_RSYNC=0
        elif [ "${OMNIVLA_DRY_RUN:-0}" != 1 ]; then
            command -v rsync > /dev/null 2>&1 || USE_RSYNC=0
            [ "$USE_RSYNC" = 0 ] || "${SSH[@]}" "$HOST" "command -v rsync" > /dev/null 2>&1 || USE_RSYNC=0
        fi
        [ "$USE_RSYNC" = 1 ] || say "rsync が無いので ssh + tar で持ってきます"
        TQ=$(printf ' %q' "${TEXCL[@]}")
        DEST=${ORIN_FETCH_DEST:-$ROOT/debug_log/orin_$(date +%Y%m%d_%H%M%S)}
        mkdir -p "$DEST"
        for d in log/field log/debug_view log/nav log/bags runs/desk_eval data/topomaps; do
            mkdir -p "$DEST/$d"
            if [ "$USE_RSYNC" = 1 ]; then
                run rsync -a -e "$RSH" "${EXCL[@]}" "$HOST:$REPO/$d/" "$DEST/$d/" 2> /dev/null || say "  ($d は無い)"
            elif [ "${OMNIVLA_DRY_RUN:-0}" = 1 ]; then
                echo "+ ${SSH[*]} $HOST cd $REPO/$d && tar cf -$TQ . | tar xf - -C $DEST/$d"
            else
                "${SSH[@]}" "$HOST" "cd $REPO/$d 2> /dev/null && tar cf -$TQ ." 2> /dev/null |
                    tar xf - -C "$DEST/$d" 2> /dev/null || say "  ($d は無い)"
            fi
        done
        find "$DEST" -type d -empty -delete 2> /dev/null || true
        say "-> $DEST ($(du -sh "$DEST" 2> /dev/null | awk '{print $1}'))"
        say "まとめ: $(ls -1 "$DEST"/log/debug_view/*/summary.txt 2> /dev/null | tail -1)"
        ;;
    ssh)
        run "${SSH[@]}" -t "$HOST" "cd $REPO && exec bash -l"
        ;;
    *) usage 1 ;;
esac
