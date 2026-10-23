#!/usr/bin/env bash
# 時刻合わせ (chrony). 単体のときはネットの時刻、本番は ATMOBI PC の時刻に Orin を合わせる (docs/jetson.md の 2-2).
# docker は使わない (ホストの chrony を設定する. sudo のパスワードを聞かれる). このファイル 1 つで動くので、
# リポジトリの無い ATMOBI PC には scp でこのファイルだけ送ればよい.
#
#   bin/timesync.sh orin       # Orin: ネットの時刻サーバ + ATMOBI PC (優先) を登録して、すぐ合わせる
#   bin/timesync.sh atmobi     # ATMOBI PC: 192.168.0.0/24 に時刻を配る (ネットに届かなくても配る)
#   bin/timesync.sh now        # すぐ合わせる (chronyc makestep)
#   bin/timesync.sh status     # 今どこに合わせているか、ずれ
#
#   ATMOBI_IP=192.168.0.100  ROBOT_SUBNET=192.168.0.0/24  (環境変数で変えられる)
#   chrony が入っていなければ apt で入れる (ネットにつながっているときに)
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
status() {
    run chronyc sources
    run chronyc tracking
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
        # conf.d を読まない古い chrony.conf なら読むようにする
        if [ "${OMNIVLA_DRY_RUN:-0}" = 1 ] || ! grep -qE "^(confdir|include) .*conf\.d" "$CONF"; then
            append_root "$CONF" "include $CONF_D/*.conf"
        fi
        run sudo mkdir -p "$CONF_D"
        # prefer: 届いているときは ATMOBI PC を優先. trust: ネットの時刻と食い違っても ATMOBI PC を正しいとみなす
        write_root "$CONF_D/atmobi.conf" "server $ATMOBI_IP iburst prefer trust"
        restart
        run sudo chronyc makestep
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
        run sudo chronyc makestep
        status
        ;;
    status)
        status
        ;;
    -h | --help | help) usage ;;
    *) usage 1 ;;
esac
