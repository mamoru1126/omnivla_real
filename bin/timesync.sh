#!/usr/bin/env bash
# 時刻合わせ (chrony). 単体のときはネットの時刻、本番は ATMOBI PC の時刻に Orin を合わせる.
# docker は使わない (ホストの chrony を設定する. sudo のパスワードを聞かれる). このファイル 1 つで動くので、
# リポジトリの無い ATMOBI PC には scp でこのファイルだけ送ればよい.
#
#   bin/timesync.sh orin       # Orin: ネットの時刻サーバ + ATMOBI PC (優先) を登録して、すぐ合わせる
#   bin/timesync.sh atmobi     # ATMOBI PC: 192.168.0.0/24 に時刻を配る (ネットに届かなくても配る)
#   bin/timesync.sh now        # すぐ合わせる (chronyc makestep. NTP に届かないときは HTTPS の応答の時刻で合わせる)
#   bin/timesync.sh status     # 今どこに合わせているか、ずれ
#   bin/timesync.sh check      # ATMOBI PC との時計のずれを 1 回測る (時計は変えない. chronyd -Q)
#   bin/timesync.sh step       # ATMOBI PC の時刻に今すぐ合わせる (chrony を一度止めて chronyd -q. 届かなければ ssh で)
#                              #   ssh で合わせる場合の相手: ATMOBI_SSH=user@192.168.0.100
#
#   ATMOBI_IP=192.168.0.100  ROBOT_SUBNET=192.168.0.0/24  (環境変数で変えられる)
#   chrony が入っていなければ apt で入れる (ネットにつながっているときに)
#   社内ネットワークなどで NTP (UDP 123) が通らないときは、HTTPS (プロキシ経由でも可) の Date ヘッダで合わせる
set -euo pipefail
ME=$(basename "$0" .sh)
ATMOBI_IP=${ATMOBI_IP:-192.168.0.100}
ROBOT_SUBNET=${ROBOT_SUBNET:-192.168.0.0/24}
CONF=${CHRONY_CONF:-/etc/chrony/chrony.conf}
CONF_D=${CHRONY_CONF_D:-/etc/chrony/conf.d}

say() { echo "[$ME] $*"; }
die() { echo "[$ME] $*" >&2; exit 1; }
usage() {
    sed -n '2,/^[^#]/{/^#/{s/^# \{0,1\}//;p}}' "$0"
    exit "${1:-0}"
}
# OMNIVLA_DRY_RUN=1: 実行せずに表示だけ
run() {
    if [ "${OMNIVLA_DRY_RUN:-0}" = 1 ]; then
        echo "+ $*"
    else
        "$@"
    fi
}
# sudo tee で書く (DRY_RUN では中身を表示)
write_root() {
    local file=$1 text=$2
    if [ "${OMNIVLA_DRY_RUN:-0}" = 1 ]; then
        echo "+ write $file:"
        printf '%s\n' "$text" | sed 's/^/    /'
    else
        printf '%s\n' "$text" | sudo tee "$file" > /dev/null
    fi
}
append_root() {
    local file=$1 line=$2
    if [ "${OMNIVLA_DRY_RUN:-0}" != 1 ] && grep -qxF -- "$line" "$file" 2> /dev/null; then return 0; fi
    if [ "${OMNIVLA_DRY_RUN:-0}" = 1 ]; then
        echo "+ append to $file: $line"
    else
        printf '%s\n' "$line" | sudo tee -a "$file" > /dev/null
    fi
}
ensure_chrony() {
    if [ "${OMNIVLA_DRY_RUN:-0}" != 1 ] && command -v chronyc > /dev/null; then return 0; fi
    say "chrony を入れます (ネットにつながっている必要があります)"
    run sudo apt-get update
    run sudo apt-get install -y chrony
}
restart() {
    run sudo systemctl restart chrony
    if [ "${OMNIVLA_DRY_RUN:-0}" != 1 ]; then sleep 3; fi
}
synced() { sudo chronyc tracking 2> /dev/null | grep -q '^Leap status *: Normal'; }
# ATMOBI PC とのずれ [s] を 1 回測る (正: この PC が遅れている). 測れなければ空
offset() {
    sudo chronyd -Q -t 15 "server $ATMOBI_IP iburst" 2>&1 | grep -oE 'wrong by -?[0-9.]+' | awk '{print $3}' | tail -1
}
# ずれの判定を表示 (0: 0.05 秒以内 / 1: それ以上 / 2: 測れない)
check() {
    if [ "${OMNIVLA_DRY_RUN:-0}" = 1 ]; then
        echo "+ sudo chronyd -Q -t 15 server $ATMOBI_IP iburst"
        return 0
    fi
    local o
    o=$(offset || true)
    if [ -z "$o" ]; then
        say "NG  ATMOBI PC ($ATMOBI_IP) の時刻を測れません (つながっていない / ATMOBI PC で bin/timesync.sh atmobi をしていない)"
        return 2
    fi
    if awk -v o="$o" 'BEGIN { exit !(o < 0.05 && o > -0.05) }'; then
        say "OK  ATMOBI PC との時計のずれ ${o} 秒"
        return 0
    fi
    say "NG  ATMOBI PC との時計のずれ ${o} 秒 (正: この PC が遅れている). 合わせる: bin/timesync.sh step"
    return 1
}
# ATMOBI PC の時刻に今すぐ合わせる
step_atmobi() {
    say "ATMOBI PC ($ATMOBI_IP) の時刻に合わせます"
    run sudo systemctl stop chrony || true
    if run sudo chronyd -q -t 20 "server $ATMOBI_IP iburst"; then
        :
    elif [ -n "${ATMOBI_SSH:-}" ]; then
        say "NTP で届かないので ssh ($ATMOBI_SSH) で ATMOBI PC の時刻を読んで合わせます (ずれは数十 ms 残る)"
        run sudo date -s "$(ssh "$ATMOBI_SSH" 'date -u +%Y-%m-%dT%H:%M:%S.%NZ')" > /dev/null
    else
        say "NTP で届きません. ssh で合わせるなら: ATMOBI_SSH=<user>@$ATMOBI_IP bin/timesync.sh step"
    fi
    run sudo systemctl start chrony || true
    check || true
}
# HTTPS の応答の Date ヘッダ (秒単位) で時計を合わせる. curl は https_proxy を使う
http_time() {
    local url d=""
    for url in https://www.google.com https://www.cloudflare.com https://github.com; do
        d=$(curl -sfI --max-time 10 "$url" 2> /dev/null | tr -d '\r' | sed -n 's/^[Dd]ate: //p' | head -1) || true
        [ -n "$d" ] && break
    done
    [ -n "$d" ] || return 1
    say "HTTPS の時刻 ($url): $d"
    run sudo date -s "$d" > /dev/null
}
# すぐ合わせる: chrony で合わせられなければ HTTPS で
step_now() {
    run sudo chronyc makestep || true
    if [ "${OMNIVLA_DRY_RUN:-0}" = 1 ]; then
        echo "+ (NTP で同期できていなければ) sudo date -s <HTTPS の Date>"
        return 0
    fi
    for _ in 1 2 3 4 5 6 7 8 9 10; do synced && return 0; sleep 1; done
    say "NTP の時刻サーバに届きません (ネットの NTP が通らない / ATMOBI PC につながっていない). HTTPS の時刻で合わせます"
    http_time || say "HTTPS にも届きません. 手で合わせてください: sudo date -s '2026-10-23 18:00:00'"
}
status() {
    run sudo chronyc sources
    run sudo chronyc tracking
    run date
}

case "${1:-}" in
    orin)
        ensure_chrony
        say "ネットの時刻サーバ + ATMOBI PC ($ATMOBI_IP, 優先) に合わせるように設定します"
        # ネットの時刻サーバ (pool / server 行) は残す (前に止めていたら戻す). ATMOBI PC の行は conf.d に置くので消す.
        # makestep 1 -1: ずれが 1 秒以上なら、起動直後に限らずいつでも一気に合わせる (単体と本番を行き来するため)
        run sudo sed -i -E -e 's/^#((pool|server) )/\1/' -e "/^server ${ATMOBI_IP//./\\.} /d" \
            -e 's/^makestep .*/makestep 1 -1/' "$CONF"
        if [ "${OMNIVLA_DRY_RUN:-0}" = 1 ] || ! grep -q '^makestep ' "$CONF"; then append_root "$CONF" "makestep 1 -1"; fi
        # ネットの時刻サーバが 1 つも書かれていなければ足す
        if [ "${OMNIVLA_DRY_RUN:-0}" = 1 ] || ! grep -qE '^(pool|server) ' "$CONF"; then
            append_root "$CONF" "pool ntp.ubuntu.com iburst maxsources 4"
        fi
        # conf.d を読まない古い chrony.conf なら読むようにする
        if [ "${OMNIVLA_DRY_RUN:-0}" = 1 ] || ! grep -qE "^(confdir|include) .*conf\.d" "$CONF"; then
            append_root "$CONF" "include $CONF_D/*.conf"
        fi
        run sudo mkdir -p "$CONF_D"
        # prefer: 届いているときは ATMOBI PC を優先. trust: ネットの時刻と食い違っても ATMOBI PC を正しいとみなす
        write_root "$CONF_D/atmobi.conf" "server $ATMOBI_IP iburst prefer trust"
        restart
        step_now
        status
        say "行頭が ^* のサーバに合わせています (単体: ネットのサーバ / 本番: $ATMOBI_IP)"
        ;;
    atmobi)
        ensure_chrony
        say "$ROBOT_SUBNET に時刻を配るように設定します (ネットに届かなくても自分の時刻を配る)"
        append_root "$CONF" "allow $ROBOT_SUBNET"
        append_root "$CONF" "local stratum 10"
        restart
        status
        say "この PC の時刻が正しいか確認してください (Orin はこの時刻に合わせます): $(date)"
        ;;
    now)
        step_now
        status
        ;;
    status)
        status
        ;;
    check)
        check
        ;;
    step)
        step_atmobi
        ;;
    -h | --help | help) usage ;;
    *) usage 1 ;;
esac
