#!/usr/bin/env bash
# 実機での走行の手順をまとめたもの (Orin の上で. ROS 2 Humble: .env の JETSON_ROS_DISTRO=humble). ネット不要.
#
#   bin/field.sh setup [--topomap 名前] # .env に走行用の値を書く (NAV_AUTOSTART=false, 画像の古さは受け取ってから, TOPOMAP)
#                                       #   ATMOBI PC のブラウザの「OmniVLA 開始」から起動した時も同じ設定になる
#   bin/field.sh topic image|odom <名前> # robot.yaml のトピック名を変える (check で NG になった時)
#   bin/field.sh check [--where]        # 走らせる前の確認 (時計・ネットワーク・重み・topomap・トピック・ノード). OK / 注意 / NG
#   bin/field.sh time                   # ATMOBI PC との時計のずれを測り、ずれていれば合わせる (sudo)
#   bin/field.sh up [--topomap 名前] [--max-v 0.3] [--max-w 0.6] [--strict-time]
#                                       # 推論サーバ + ナビゲーション + デバッグ画面を起動して、準備ができるまで待つ (走り出さない)
#   bin/field.sh where [topomap]        # いまのカメラ画像が topomap の何番に似ているか (スタート位置・カメラの確認)
#   bin/field.sh start | stop           # 走行開始 / 停止 (/omnivla/enable). start は数秒後の状態も表示
#   bin/field.sh status                 # 状態を 1 回
#   bin/field.sh record [名前] | record stop | record status   # bag の記録 (裏で. log/bags/<名前>_<時刻>)
#   bin/field.sh topomap <bag> <名前> [--spacing 1.0]           # 録った bag から topomap を作る (data/topomaps/<名前>)
#   bin/field.sh eval <bag> [topomap]   # 録った bag で机上評価 (runs/desk_eval/). 推論サーバを使う
#   bin/field.sh summary                # デバッグ画面のまとめ (一番新しいもの)
#   bin/field.sh collect [出力先]        # 持ち帰るログを 1 つにまとめる (bag と画像は除く. 既定 log/collect/<時刻>.tar.gz)
#   bin/field.sh update <bundle>        # 作業 PC から運んだ更新 (bin/orin.sh bundle) を取り込む
#   bin/field.sh down [--all]           # ナビゲーションとデバッグ画面と記録を止める (--all で推論サーバも)
#
# 画像の古さ: up は既定で「受け取ってから」で判定する (engine.image_age_from=receive). 2 台の時計のずれや
#   カメラの時刻の付け方で止まらない. --strict-time で画像の時刻 (header.stamp) で判定する (時計が合っている時だけ).
# デバッグ画面: http://<Orin の IP>:8090 (ATMOBI PC から http://192.168.0.200:8090)
source "$(dirname "$0")/_lib.sh"

DISTRO=$(env_get JETSON_ROS_DISTRO noetic)
PORT=$(env_get POLICY_PORT 8765)
ATMOBI_IP=${ATMOBI_IP:-$(env_get ATMOBI_IP 192.168.0.100)}
export ATMOBI_IP   # bin/timesync.sh も同じ相手を見る
SETUP="source /opt/omnivla_ws/install/setup.bash"
DEBUG_NAME=omnivla_debug_view
RECORD_NAME=omnivla_record

need_humble() {
    [ "$DISTRO" = humble ] || die "ROS 2 (JETSON_ROS_DISTRO=humble) 用です (今: $DISTRO)"
}
# nav のイメージでコマンド (nav が動いていなくてよい. 同じ ROS_DOMAIN_ID / DDS の設定)
nav_run() { jetson run --rm --no-deps "${TTY[@]}" nav bash -c "$SETUP && $*"; }
running() { [ -n "$(jetson ps --status running -q "$1" 2> /dev/null)" ]; }
cont_running() { [ "$(command docker inspect -f '{{.State.Running}}' "$1" 2> /dev/null)" = true ]; }
policy_info() {
    if command -v curl > /dev/null; then
        curl -sf --noproxy '*' --max-time 3 "http://127.0.0.1:$PORT/info"
    else
        python3 -c "import urllib.request,sys; sys.stdout.write(urllib.request.urlopen('http://127.0.0.1:$PORT/info', timeout=3).read().decode())"
    fi
}
# robot.yaml の topics.<key>
robot_yaml() { hpath "$(env_get ROBOT_CONFIG /workspace/configs/robot.yaml)"; }
topic_of() { sed -n -E "s/^[[:space:]]+$1:[[:space:]]*\"?([^\" #]+)\"?.*/\\1/p" "$(robot_yaml)" | head -1; }
mark() { printf '[%-2s] %s\n' "$1" "$2"; }
# topomap のフォルダの画像 (0.jpg, 1.jpg, ...) の枚数
count_nodes() {
    local n=0 f
    for f in "$1"/[0-9]*.jpg; do [ -e "$f" ] && n=$((n + 1)); done
    echo "$n"
}
NG_COUNT=0
ng() { mark NG "$1"; NG_COUNT=$((NG_COUNT + 1)); [ -z "${2:-}" ] || echo "        $2"; }
ok() { mark OK "$1"; }
warn() { mark 注意 "$1"; [ -z "${2:-}" ] || echo "        $2"; }

# bag (名前 / log/bags/... / パス) → ホストのパス. 記録が途中で切れて metadata.yaml が無ければ作り直す
bag_host() {
    local b=$1 h
    for h in "$b" "$CALLER_PWD/$b" "$ROOT/log/bags/$b" "$ROOT/$b"; do
        if [ -e "$h" ]; then
            b=$(realpath "$h")
            break
        fi
    done
    [ -e "$b" ] || die "bag が見つかりません: $1 (log/bags/ の中: $(ls "$ROOT/log/bags" 2> /dev/null | tr '\n' ' '))"
    [[ $b == "$ROOT"/* ]] || [[ $b == "$(host_dir bags)"/* ]] || die "$b はコンテナから見えません (リポジトリの中か bags/ に置く)"
    if [ -d "$b" ] && [ ! -f "$b/metadata.yaml" ] && [ "${OMNIVLA_DRY_RUN:-0}" != 1 ]; then
        say "metadata.yaml が無いので作り直します (記録が途中で切れた bag): ros2 bag reindex"
        nav_run "ros2 bag reindex $(cpath bags "" "$b")" >&2
    fi
    echo "$b"
}
# その bag 用の robot.yaml (トピックを bag の中身に合わせたもの)
bag_robot() {
    local cb=$1
    in_shell python3 tools/bag_robot_yaml.py "$cb" >&2
    echo "${cb%/}.robot.yaml"
}

CMD=${1:-}
shift || true
# .env の KEY を VALUE にする (無ければ足す). 元は .env.bak_<時刻>
set_env() {
    local k=$1 v=$2
    if [ "${OMNIVLA_DRY_RUN:-0}" = 1 ]; then
        echo "+ .env: $k=$v"
        return
    fi
    if grep -qE "^[[:space:]]*(export[[:space:]]+)?$k=" "$ROOT/.env"; then
        local tmp
        tmp=$(mktemp)
        awk -v k="$k" -v v="$v" 'BEGIN { re = "^[[:space:]]*(export[[:space:]]+)?" k "=" }
            $0 ~ re { print k "=" v; next } { print }' "$ROOT/.env" > "$tmp" && cat "$tmp" > "$ROOT/.env" && rm -f "$tmp"
    else
        printf '%s=%s\n' "$k" "$v" >> "$ROOT/.env"
    fi
    say ".env: $k=$v"
}

case $CMD in
    setup)
        ensure_env
        [ "${OMNIVLA_DRY_RUN:-0}" = 1 ] || cp "$ROOT/.env" "$ROOT/.env.bak_$(date +%Y%m%d_%H%M%S)"
        while [ $# -gt 0 ]; do
            case $1 in
                --topomap) set_env TOPOMAP "$(topomap_path "$2")"; shift ;;
                *) die "知らないオプション: $1" ;;
            esac
            shift
        done
        set_env NAV_AUTOSTART false
        OV=$(env_get NAV_OVERRIDES "")
        for kv in engine.image_age_from=receive engine.max_image_age=1.0; do
            [[ " $OV " == *" ${kv%%=*}="* ]] || OV="$OV $kv"
        done
        set_env NAV_OVERRIDES "\"$(echo "$OV" | xargs)\""
        say "元の .env は .env.bak_* に残しています. 反映: bin/field.sh up"
        ;;
    topic)
        [ $# -eq 2 ] || die "使い方: bin/field.sh topic image|odom <トピック名>"
        case $1 in image | odom | cmd) ;; *) die "image か odom か cmd" ;; esac
        RY=$(robot_yaml)
        [ "${OMNIVLA_DRY_RUN:-0}" = 1 ] || cp "$RY" "$RY.bak_$(date +%Y%m%d_%H%M%S)"
        if [ "${OMNIVLA_DRY_RUN:-0}" = 1 ]; then
            echo "+ $RY: topics.$1: $2"
        else
            sed -i -E "s|^([[:space:]]+$1:[[:space:]]*)[^ #]*|\1$2|" "$RY"
        fi
        say "$RY: topics.$1 = $(topic_of "$1")  (反映: bin/field.sh up)"
        ;;
    check)
        WHERE=()
        [ "${1:-}" = --where ] && WHERE=(--where)
        echo "== Orin"
        if ip -br addr 2> /dev/null | grep -q '192\.168\.0\.'; then ok "192.168.0.x のアドレス: $(hostname -I 2> /dev/null)"; else warn "192.168.0.x のアドレスが無い ($(hostname -I 2> /dev/null))" "ロボットの LAN につながっているか確認"; fi
        if ping -c 1 -W 1 "$ATMOBI_IP" > /dev/null 2>&1; then ok "ATMOBI PC ($ATMOBI_IP) に ping が通る"; else ng "ATMOBI PC ($ATMOBI_IP) に ping が通らない" "ケーブル / IP の設定"; fi
        if bash bin/timesync.sh check; then :; else warn "時計: bin/field.sh time で合わせる (up は既定で時計がずれていても止まらない)"; fi
        A=$(df -Ph "$ROOT" | awk 'NR==2 {print $4}')
        ok "ディスクの空き $A"
        command -v nvpmodel > /dev/null && echo "        電源モード: $(nvpmodel -q 2> /dev/null | head -1)"
        echo "== 設定 (.env)"
        [ "$DISTRO" = humble ] && ok "JETSON_ROS_DISTRO=humble" || warn "JETSON_ROS_DISTRO=$DISTRO"
        M=$(env_get NAV_MODEL "")
        F=$(env_get FINETUNED_DIR "")
        W=$(env_get NAV_WEIGHTS "")
        if [ -n "$F" ] && [ -e "$(hpath "$F")" ]; then ok "NAV_MODEL=$M  FINETUNED_DIR=$F"; elif [ -n "$F" ]; then ng "FINETUNED_DIR が無い: $F" "学習 PC から bin/to_jetson.sh で送る"; else warn "FINETUNED_DIR が空 (公式の重みで動く)"; fi
        if [ "$M" = 7b ]; then
            B=${W:-/checkpoints/omnivla-original}
            [ -e "$(hpath "$B")" ] && ok "7B の元のモデル $B" || ng "7B の元のモデルが無い: $B ($(hpath "$B"))"
        fi
        T=$(env_get TOPOMAP "")
        if [ -z "$T" ]; then
            warn "TOPOMAP が空" "bin/field.sh up --topomap <名前> で指定 (data/topomaps: $(ls "$(host_dir data)/topomaps" 2> /dev/null | tr '\n' ' '))"
        elif [ -d "$(hpath "$T")" ]; then
            ok "TOPOMAP=$T ($(count_nodes "$(hpath "$T")") 枚)"
        else
            ng "TOPOMAP が無い: $T"
        fi
        D=$(env_get DDS_PROFILE_FILE ../nakanoshima_2026/docker/fastdds_robot_links.xml)
        case $D in /*) DH=$D ;; *) DH=$ROOT/$D ;; esac
        if [ "$DISTRO" = humble ]; then
            if [ -f "$DH" ]; then
                ADDR=$(grep -oE '<address>[^<]+</address>' "$DH" | sed -E 's/<\/?address>//g' | grep -v '^127\.' | tr '\n' ' ' || true)
                MINE=$(hostname -I 2> /dev/null || true)
                HIT=""
                for a in $ADDR; do [[ " $MINE " == *" $a "* ]] && HIT="$HIT $a"; done
                if [ -z "$ADDR" ] || [ -n "$HIT" ]; then ok "DDS の設定 $D${HIT:+ (この PC のアドレス:$HIT)}"; else ng "DDS の設定にあるアドレス ($ADDR) がこの PC に無い" "Orin の IP か $D を確認 (通信できない)"; fi
            else
                ng "DDS の設定ファイルが無い: $DH" ".env の DDS_PROFILE_FILE"
            fi
        fi
        echo "== イメージ・コンテナ"
        for img in omnivla_real:jetson-policy "omnivla_real:$([ "$DISTRO" = humble ] && echo ros2-humble || echo ros1)"; do
            if command docker image inspect "$img" > /dev/null 2>&1; then ok "イメージ $img"; else ng "イメージ $img が無い" "ネットにつながる時に bin/jetson.sh build"; fi
        done
        if running policy; then
            I=$(policy_info 2> /dev/null || true)
            if [ -n "$I" ]; then ok "推論サーバ: $(echo "$I" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("model"), d.get("checkpoint",""))' 2> /dev/null)"; else warn "推論サーバは起動中 (モデルの読み込み中. 7B は数分)"; fi
        else
            warn "推論サーバは止まっている" "bin/field.sh up"
        fi
        running nav && ok "ナビゲーション (nav) は動いている" || warn "ナビゲーション (nav) は止まっている" "bin/field.sh up"
        cont_running $DEBUG_NAME && ok "デバッグ画面 http://$(hostname -I | awk '{print $1}'):8090" || true
        cont_running $RECORD_NAME && warn "bag を記録中 (bin/field.sh record stop で止める)" || true
        if [ "$DISTRO" = humble ]; then
            echo "== ROS 2 のトピック (数秒受け取る)"
            nav_run "python3 /workspace/tools/field_probe.py --sec 5 ${WHERE[*]:-}" || NG_COUNT=$((NG_COUNT + 1))
        fi
        echo
        if [ "$NG_COUNT" = 0 ]; then say "NG はありません"; else say "NG が $NG_COUNT 件あります (上の NG の行を確認)"; fi
        ;;
    time)
        if bash bin/timesync.sh check; then
            say "時計は合っています"
        else
            bash bin/timesync.sh step
        fi
        ;;
    up)
        need_humble
        ensure_env
        TOPO=$(env_get TOPOMAP "")
        STRICT=0
        EXTRA=()
        while [ $# -gt 0 ]; do
            case $1 in
                --topomap) TOPO=$2; shift ;;
                --max-v) EXTRA+=("engine.controller.track_max_v=$2"); shift ;;
                --max-w) EXTRA+=("engine.controller.track_max_w=$2"); shift ;;
                --strict-time) STRICT=1 ;;
                *) die "知らないオプション: $1 (bin/field.sh -h)" ;;
            esac
            shift
        done
        [ -n "$TOPO" ] || die "topomap を指定してください: bin/field.sh up --topomap <名前> (data/topomaps: $(ls "$(host_dir data)/topomaps" 2> /dev/null | tr '\n' ' '))"
        TOPO=$(topomap_path "$TOPO")
        # .env の NAV_OVERRIDES に、画像の古さの測り方と速度の上限を足す (同じキーは後ろが勝つので、ここで決めたものを優先)
        OV=""
        read -r -a KVS <<< "$(env_get NAV_OVERRIDES "")"
        for kv in "${KVS[@]}"; do
            case ${kv%%=*} in engine.image_age_from | engine.max_image_age) ;; *) OV="$OV $kv" ;; esac
        done
        if [ "$STRICT" = 0 ]; then
            OV="$OV engine.image_age_from=receive engine.max_image_age=1.0"
        else
            OV="$OV engine.image_age_from=stamp"
        fi
        OV=$(echo "$OV ${EXTRA[*]:-}" | xargs)
        export TOPOMAP=$TOPO NAV_AUTOSTART=false NAV_OVERRIDES=$OV
        say "topomap: $TOPO ($(count_nodes "$(hpath "$TOPO")") 枚)  モデル: $(env_get NAV_MODEL '') $(env_get FINETUNED_DIR '')"
        say "NAV_OVERRIDES: ${OV:-(なし)}"
        jetson up -d policy nav
        if [ "${OMNIVLA_DRY_RUN:-0}" != 1 ]; then
            say "推論サーバの準備を待っています (7B は数分)"
            T0=$SECONDS
            until policy_info > /dev/null 2>&1; do
                running policy || die "推論サーバが止まりました: bin/jetson.sh logs policy"
                if (( (SECONDS - T0) % 30 == 0 )); then say "  $((SECONDS - T0)) 秒: $(jetson logs --tail 1 --no-log-prefix policy 2> /dev/null | tail -1)"; fi
                (( SECONDS - T0 < 1800 )) || die "30 分待っても推論サーバが応答しません: bin/jetson.sh logs policy"
                sleep 1
            done
            say "推論サーバ OK ($((SECONDS - T0)) 秒). ナビゲーションの準備を待っています"
            for _ in $(seq 120); do
                jetson logs --no-log-prefix nav 2> /dev/null | grep -q "ready:" && break
                running nav || die "ナビゲーションが止まりました: bin/jetson.sh logs nav"
                sleep 1
            done
            jetson logs --no-log-prefix nav 2> /dev/null | grep -E "ready:|NAV_OVERRIDES|ERROR" | tail -3 || true
        fi
        # デバッグ画面 (別のコンテナで. 止めるとまとめを書く)
        command docker rm -f "$DEBUG_NAME" > /dev/null 2>&1 || true
        jetson run -d --rm --no-deps --name "$DEBUG_NAME" nav bash -c "$SETUP && exec python3 /workspace/tools/debug_view.py" > /dev/null
        say "トピックとノードの確認 (数秒):"
        nav_run "python3 /workspace/tools/field_probe.py --sec 3" || true
        say "準備できました. デバッグ画面: http://$(hostname -I 2> /dev/null | awk '{print $1}'):8090  (ATMOBI PC から http://192.168.0.200:8090)"
        say "次: bin/field.sh where (スタート位置の確認) → bin/field.sh start"
        ;;
    where)
        need_humble
        T=""
        [ -n "${1:-}" ] && T=$(topomap_path "$1")   # 省くと動いているノードの topomap (無ければ .env の TOPOMAP)
        running policy || die "推論サーバが動いていません: bin/field.sh up"
        nav_run "python3 /workspace/tools/field_probe.py --sec 2 --where ${T:+--topomap $T}"
        ;;
    start | stop)
        need_humble
        running nav || die "ナビゲーションが動いていません: bin/field.sh up"
        V=$([ "$CMD" = start ] && echo true || echo false)
        nav_run "python3 /workspace/tools/field_probe.py --enable $V"
        ;;
    status)
        need_humble
        nav_run "python3 /workspace/tools/field_probe.py --status"
        ;;
    record)
        need_humble
        case "${1:-}" in
            stop)
                cont_running $RECORD_NAME || die "記録していません"
                command docker stop -t 20 "$RECORD_NAME" > /dev/null   # SIGINT (compose の stop_signal) で bag を閉じる
                L=$(ls -1dt "$ROOT"/log/bags/*/ 2> /dev/null | head -1 || true)
                say "記録を止めました: ${L%/}"
                [ -n "$L" ] && du -sh "${L%/}" 2> /dev/null | awk '{print "  大きさ " $1}'
                exit 0
                ;;
            status)
                cont_running $RECORD_NAME && say "記録中: $(ls -1dt "$ROOT"/log/bags/*/ 2> /dev/null | head -1 || true)" || say "記録していません"
                exit 0
                ;;
        esac
        cont_running $RECORD_NAME && die "もう記録しています (bin/field.sh record stop)"
        NAME=${1:-run}
        OUT=/workspace/log/bags/${NAME}_$(date +%Y%m%d_%H%M%S)
        mkdir -p "$ROOT/log/bags"
        TOPICS=$(printf '%s ' "$(topic_of image)" "$(topic_of odom)" /cmd_vel "$(env_get CMD_VEL_TOPIC /omnivla/cmd_vel/raw)" \
            /omnivla/status /omnivla/path | xargs -n1 | awk 'NF && !seen[$0]++' | xargs)
        command docker rm -f "$RECORD_NAME" > /dev/null 2>&1 || true
        jetson run -d --rm --no-deps --name "$RECORD_NAME" nav bash -c "$SETUP && exec ros2 bag record -o $OUT $TOPICS" > /dev/null
        say "記録を始めました: $(hpath "$OUT")"
        say "トピック: $TOPICS"
        say "止める: bin/field.sh record stop"
        ;;
    topomap)
        [ $# -ge 2 ] || die "使い方: bin/field.sh topomap <bag> <名前> [--spacing 1.0]"
        B=$(bag_host "$1")
        NAME=$2
        shift 2
        CB=$(cpath bags "" "$B")
        RY=$(bag_robot "$CB")
        bash bin/make_topomap.sh "$NAME" "$B" --robot "$RY" "$@"
        say "使うとき: bin/field.sh up --topomap $NAME"
        ;;
    eval)
        [ $# -ge 1 ] || die "使い方: bin/field.sh eval <bag> [topomap]"
        B=$(bag_host "$1")
        shift
        T=${1:-$(env_get TOPOMAP "")}
        [ $# -gt 0 ] && shift
        [ -n "$T" ] || die "topomap を指定してください: bin/field.sh eval <bag> <topomap>"
        CB=$(cpath bags "" "$B")
        RY=$(bag_robot "$CB")
        bash bin/desk_eval.sh "$B" "$T" --robot "$RY" "$@"
        ;;
    summary)
        L=$ROOT/log/debug_view/latest
        [ -e "$L" ] || L=$(ls -1dt "$ROOT"/log/debug_view/*/ 2> /dev/null | head -1 || true)
        [ -n "$L" ] && [ -e "$L/steps.csv" ] || die "デバッグ画面の記録がありません (log/debug_view)"
        in_shell python3 tools/debug_view.py --summarize "$(cpath bags "" "$(realpath "$L")")"
        ;;
    collect)
        DEST=$ROOT/log/collect
        [ -n "${1:-}" ] && DEST=$(cd "$CALLER_PWD" && realpath -m -- "$1")
        mkdir -p "$DEST"
        F=$DEST/orin_$(date +%Y%m%d_%H%M%S).tar.gz
        LIST=()
        for d in log/field log/debug_view log/nav runs/desk_eval data/topomaps log/bags; do [ -e "$d" ] && LIST+=("$d"); done
        # 画像 (debug_view の frames, nav の raw, desk_eval の debug) と bag 本体と topomap の画像は除く (大きいので. 必要なら bin/orin.sh fetch)
        tar -czf "$F" --exclude='*/frames' --exclude='*/raw' --exclude='*/goals' --exclude='*/debug' \
            --exclude='*.db3' --exclude='*.mcap' --exclude='data/topomaps/*/*.jpg' "${LIST[@]}" 2> /dev/null || true
        say "-> $F ($(du -h "$F" | awk '{print $1}'))"
        ;;
    update)
        [ $# -ge 1 ] || die "使い方: bin/field.sh update <bundle> (作業 PC で bin/orin.sh bundle)"
        BR=$(git rev-parse --abbrev-ref HEAD)
        F=$(cd "$CALLER_PWD" && realpath -- "$1")
        git fetch "$F" "$BR"
        git merge --ff-only FETCH_HEAD
        git log --oneline -3
        say "更新しました. 動いているノードに反映するには: bin/field.sh up (イメージの作り直しは不要)"
        ;;
    down)
        for c in "$DEBUG_NAME" "$RECORD_NAME"; do
            if cont_running "$c"; then command docker stop -t 20 "$c" > /dev/null && say "止めました: $c"; fi
        done
        if [ "${1:-}" = --all ]; then
            jetson down
        else
            jetson stop nav
            say "推論サーバはそのまま (止める: bin/field.sh down --all)"
        fi
        ;;
    *) usage 1 ;;
esac
