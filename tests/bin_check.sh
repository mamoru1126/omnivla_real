#!/usr/bin/env bash
# bin/*.sh が引数とパスを正しく docker のコマンドに直すかの確認 (docker は実行しない: OMNIVLA_DRY_RUN=1).
#   bash tests/bin_check.sh
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
mkdir -p "$T/bags/split" "$T/bags/ros2run" "$T/data/topomaps/course_a" "$T/runs/r1/checkpoints/step_000010" \
    "$T/runs/replay/nav" "$T/checkpoints/omnivla-edge" "$T/elsewhere"
touch "$T/bags/run1.bag" "$T/bags/split/a_0.bag" "$T/bags/split/a_1.bag" "$T/elsewhere/x.bag"
export OMNIVLA_DRY_RUN=1 OMNIVLA_JETSON=0 BAG_DIR=$T/bags DATA_DIR=$T/data RUNS_DIR=$T/runs CHECKPOINT_DIR=$T/checkpoints
unset NAV_MODEL NAV_WEIGHTS FINETUNED_DIR TOPOMAP REPLAY_BAG
B=$ROOT/bin
NG=0

# check "期待する文字列" コマンド...   (出力に含まれていれば OK)
check() {
    local want=$1 out
    shift
    if ! out=$("$@" 2>&1); then
        echo "NG (exit): $*"
        echo "$out" | sed 's/^/    /'
        NG=1
        return
    fi
    if grep -qF -- "$want" <<< "$out"; then
        echo "ok  $*"
    else
        echo "NG: $*"
        echo "    want: $want"
        echo "$out" | sed 's/^/    /'
        NG=1
    fi
}
# fails "期待するエラー" コマンド...   (失敗して、出力に含まれていれば OK)
fails() {
    local want=$1 out
    shift
    if out=$("$@" 2>&1); then
        echo "NG (should fail): $*"
        NG=1
    elif grep -qF -- "$want" <<< "$out"; then
        echo "ok  (fails) $*"
    else
        echo "NG: $*"
        echo "    want: $want"
        echo "$out" | sed 's/^/    /'
        NG=1
    fi
}

# --- bag の指定のしかた ---
check "shell python3 tools/bag_info.py /bags/run1.bag" "$B/bag_info.sh" run1.bag
check "tools/bag_info.py /bags/run1.bag" "$B/bag_info.sh" "$T/bags/run1.bag"
check "tools/bag_info.py /bags/run1.bag" "$B/bag_info.sh" /bags/run1.bag
check "tools/bag_info.py /bags/split/a_0.bag,/bags/split/a_1.bag" "$B/bag_info.sh" split/a_0.bag,split/a_1.bag
check "tools/bag_info.py /bags/ros2run" "$B/bag_info.sh" ros2run
check "tools/bag_info.py /bags/run1.bag" bash -c "cd '$T/bags' && '$B/bag_info.sh' ./run1.bag"
fails "見つかりません" "$B/bag_info.sh" nope.bag
fails "コンテナから見えません" "$B/bag_info.sh" "$T/elsewhere/x.bag"

# --- 各段階 ---
check "check_odometry.py --out /runs/check/run1 /bags/run1.bag --start_sec 5" "$B/check_odometry.sh" run1.bag --start_sec 5
check "tools/bag_to_dataset.py --out \"\$0\" \"\$@\"" "$B/make_dataset.sh" run1.bag split/a_0.bag,split/a_1.bag --pose_source cmd
check "/data/dataset /bags/run1.bag /bags/split/a_0.bag,/bags/split/a_1.bag --pose_source cmd" "$B/make_dataset.sh" run1.bag split/a_0.bag,split/a_1.bag --pose_source cmd
check "train_7b python3 training/finetune_omnivla.py --config configs/finetune_7b.yaml --max_steps 5" "$B/train.sh" 7b --max_steps 5
check "-d --name omnivla_train_edge train_edge python3 training/finetune_edge.py --config configs/finetune_edge.yaml" "$B/train.sh" edge -d
fails "bin/train.sh 7b" "$B/train.sh" 13b
check "make_topomap.py --out /data/topomaps/course_b /bags/run1.bag --spacing 2" "$B/make_topomap.sh" course_b run1.bag --spacing 2
check "desk_eval.py --bag /bags/run1.bag --out /runs/desk_eval/run1_" "$B/desk_eval.sh" run1.bag course_a
check "--topomap /data/topomaps/course_a --model edge --finetuned_dir /runs/r1/checkpoints/step_000010 --policy oracle" \
    env NAV_MODEL=edge FINETUNED_DIR=/runs/r1/checkpoints/step_000010 "$B/desk_eval.sh" run1.bag course_a --policy oracle
check "--topomap /data/topomaps/course_a" env TOPOMAP=/data/topomaps/course_a "$B/desk_eval.sh" run1.bag
fails "見つかりません" "$B/desk_eval.sh" run1.bag course_zzz
check "run_eval.py --bag /bags/run1.bag --topomap /data/topomaps/course_a --out /runs/run_eval/run1_" "$B/run_eval.sh" run1.bag course_a
check "finetune_omnivla.py --config configs/finetune_7b.yaml --run_name omnivla7b_overnight_" "$B/overnight.sh" run1.bag
check '--bag_weights {"run1": 3}' "$B/overnight.sh" run1.bag
check '--bag_weights {"run1": 5}' "$B/overnight.sh" run1.bag --weight 5 --course course_x
check "tools/bag_info.py /bags/run1.bag --robot configs/robot.yaml" "$B/overnight.sh" run1.bag
check "desk_eval_b_new_course_original_model" env FINETUNED_DIR=/runs/r1/checkpoints/step_000010 "$B/overnight.sh" run1.bag
fails "知らないオプション" "$B/overnight.sh" run1.bag --nope
check "--goal_mode hindsight --pose_source odom --finetuned_dir /runs/r1/checkpoints/step_000010" \
    "$B/turn_check.sh" run1.bag --new /runs/r1/checkpoints/step_000010
check "5_new_course_topomap_strict_new" env FINETUNED_DIR=/runs/r1/checkpoints/step_000010 TURN_CHECK_COURSE=course_a \
    "$B/turn_check.sh" run1.bag --new /runs/r1/checkpoints/step_000010
check "--loc_topic /localization" "$B/run_eval.sh" run1.bag course_a --loc_topic /localization

# --- 学習 PC で実機と同じ構成 ---
check "up -d --force-recreate --no-deps nav record robot" "$B/replay.sh" run1.bag course_a --rate 2 --manual
check "bag: /bags/split/a_0.bag,/bags/split/a_1.bag  topomap: /data/topomaps/course_a" "$B/replay.sh" split/a_0.bag,split/a_1.bag course_a
check "docker compose -f docker-compose.replay.yml logs -f --tail 50 robot nav" "$B/replay.sh" run1.bag course_a
check "docker compose -f docker-compose.replay.yml restart nav" "$B/replay.sh" again
check "docker compose -f docker-compose.replay.yml down" "$B/replay.sh" down
fails "topomap を指定" "$B/replay.sh" run1.bag
fails "知らないオプション" "$B/replay.sh" run1.bag course_a --fast
check "tools/plot_nav_log.py /runs/replay/nav/latest" "$B/plot_nav_log.sh"
check "up -d --force-recreate --no-deps nav record robot" "$B/replay_video.sh" run1.bag course_a --rate 2
check "robot python3 tools/nav_video.py /runs/replay/nav/DRYRUN --out /runs/replay/DRYRUN.mp4 --speed 4" \
    "$B/replay_video.sh" run1.bag course_a --speed 4
mkdir -p "$T/runs/replay/nav/20261013_100000" "$T/runs/replay/nav/20261013_110000"
check "tools/nav_video.py /runs/replay/nav/20261013_100000 --out /runs/replay/nav/20261013_100000/video.mp4" \
    "$B/replay_video.sh" --log "$T/runs/replay/nav/20261013_100000"
check "tools/nav_video.py /runs/replay/nav/20261013_110000 --out /runs/replay/nav/20261013_110000/video.mp4" \
    "$B/replay_video.sh" --log /runs/replay/nav/latest
check "--out /runs/videos/a.mp4 --fps 5" "$B/replay_video.sh" --log /runs/replay/nav/latest --out "$T/runs/videos/a.mp4" --fps 5
fails "コンテナから見えない" "$B/replay_video.sh" --log /runs/replay/nav/latest --out /tmp/x.mp4
fails "知らないオプション" "$B/replay_video.sh" run1.bag course_a --fast

# --- Jetson ---
check "rsync -aR --info=progress2 $T/runs/./r1/checkpoints/step_000010 me@jet:omnivla_real/runs/" \
    env NAV_MODEL=edge FINETUNED_DIR=/runs/r1/checkpoints/step_000010 TOPOMAP=course_a "$B/to_jetson.sh" me@jet
check "rsync -aR --info=progress2 $T/data/./topomaps/course_a me@jet:/opt/o/data/" \
    env NAV_MODEL=edge TOPOMAP=course_a "$B/to_jetson.sh" me@jet --dest /opt/o
check "$T/checkpoints/./omnivla-edge me@jet:omnivla_real/checkpoints/" env NAV_MODEL=edge TOPOMAP=course_a "$B/to_jetson.sh" me@jet
check "configs/./robot.yaml configs/./navigator.yaml me@jet:omnivla_real/configs/" env TOPOMAP=course_a "$B/to_jetson.sh" me@jet --configs
check "docker compose -f docker-compose.jetson.yml --profile standalone up -d" env TOPOMAP=/data/topomaps/course_a "$B/jetson.sh" up --standalone
check "docker compose -f docker-compose.jetson.yml up -d" env TOPOMAP=/data/topomaps/course_a "$B/jetson.sh" up
check "nav python3 tools/plot_nav_log.py /workspace/log/nav/latest" "$B/jetson.sh" plot
check "docker compose -f docker-compose.jetson.yml --profile tools run --rm --no-deps -T shell rostopic hz /odom" \
    env ROS_MASTER_URI=http://192.168.0.100:11311 ROS_IP=192.168.0.200 "$B/jetson.sh" shell rostopic hz /odom
check "ROS master: http://192.168.0.100:11311  ROS_IP: 192.168.0.200" \
    env ROS_MASTER_URI=http://192.168.0.100:11311 ROS_IP=192.168.0.200 "$B/jetson.sh" shell rostopic list
check "ROS master: http://localhost:11311  ROS_IP: 127.0.0.1" \
    env ROS_MASTER_URI=http://192.168.0.100:11311 ROS_IP=192.168.0.200 "$B/jetson.sh" shell --standalone rostopic list
check "Jetson 単体 (ROS master: http://localhost:11311)" \
    env TOPOMAP=/data/topomaps/course_a ROS_MASTER_URI=http://192.168.0.100:11311 "$B/jetson.sh" up --standalone
check "run --rm --no-deps -T shell bash" "$B/jetson.sh" shell
check "docker compose -f docker-compose.jetson.yml --profile standalone down" "$B/jetson.sh" down

# --- Jetson の上で、学習 PC と同じ確認 (OMNIVLA_JETSON=1 で Jetson のつもりにする) ---
J=(env OMNIVLA_JETSON=1)
check "docker compose -f docker-compose.jetson.yml --env-file .env.jetson up -d policy" "${J[@]}" "$B/desk_eval.sh" run1.bag course_a
check "docker compose -f docker-compose.jetson.yml --env-file .env.jetson run --rm --no-deps -T policy python3 tools/desk_eval.py --bag /bags/run1.bag" \
    "${J[@]}" "$B/desk_eval.sh" run1.bag course_a
check "--topomap /data/topomaps/course_a --model remote --url http://127.0.0.1:8765" \
    "${J[@]}" NAV_MODEL=edge FINETUNED_DIR=/runs/r1/checkpoints/step_000010 "$B/desk_eval.sh" run1.bag course_a
check "--model 7b --finetuned_dir /runs/omnivla7b_20261009_134144/checkpoints/step_005000 --policy oracle" "${J[@]}" "$B/desk_eval.sh" run1.bag course_a --policy oracle
check "jetson.yml --env-file .env.jetson run --rm --no-deps -T policy python3 tools/bag_info.py /bags/run1.bag" "${J[@]}" "$B/bag_info.sh" run1.bag
check "jetson.yml --env-file .env.jetson run --rm --no-deps -T policy python3 tools/make_topomap.py --out /data/topomaps/course_b" \
    "${J[@]}" "$B/make_topomap.sh" course_b run1.bag
check "Jetson で bag を流します" "${J[@]}" "$B/replay.sh" run1.bag course_a
check "up -d --force-recreate --no-deps nav record robot" "${J[@]}" "$B/replay_video.sh" run1.bag course_a
check "docker-compose.jetson.yml" "${J[@]}" bash -c "source '$B/_lib.sh'; echo \$REPLAY_POLICY_COMPOSE"
check "ROS master: http://192.168.0.100:11311" "${J[@]}" "$B/jetson.sh" shell rostopic list   # .env.jetson を読む
check "docker compose -f docker-compose.jetson.yml --env-file .env.jetson build" "${J[@]}" "$B/setup.sh" none
fails "学習 PC で" "${J[@]}" "$B/train.sh" 7b
fails "学習 PC で" "${J[@]}" "$B/make_dataset.sh" run1.bag
check "rsync -aR --info=progress2 $T/bags/./run1.bag me@jet:omnivla_real/bags/" \
    env TOPOMAP=course_a "$B/to_jetson.sh" me@jet --bag run1.bag
if command -v docker > /dev/null && docker compose version > /dev/null 2>&1; then
    # Jetson では bag 用の構成の推論サーバが Jetson 用のものになる
    check "image: omnivla_real:jetson-policy" bash -c "cd '$ROOT' && REPLAY_POLICY_COMPOSE=docker-compose.jetson.yml docker compose -f docker-compose.replay.yml config policy"
    check "image: omnivla_real:latest" bash -c "cd '$ROOT' && docker compose -f docker-compose.replay.yml config policy"
fi

# --- 時刻合わせ ---
check "+ write /etc/chrony/conf.d/atmobi.conf:" "$B/timesync.sh" orin
check "server 192.168.0.100 iburst prefer trust" "$B/timesync.sh" orin
check "append to /etc/chrony/chrony.conf: allow 192.168.0.0/24" "$B/timesync.sh" atmobi
check "server 10.0.0.5 iburst prefer trust" env ATMOBI_IP=10.0.0.5 "$B/timesync.sh" orin
check "+ sudo chronyc makestep" "$B/timesync.sh" now
check "sudo date -s <HTTPS の Date>" "$B/timesync.sh" now
check "append to /etc/chrony/chrony.conf: pool ntp.ubuntu.com iburst maxsources 4" "$B/timesync.sh" orin
fails "bin/timesync.sh" "$B/timesync.sh" bogus

# --- 使い方の表示 ---
for f in "$B"/*.sh; do
    [ "$(basename "$f")" = _lib.sh ] && continue
    check "bin/$(basename "$f")" "$f" -h
done

[ $NG = 0 ] && echo "all OK" || echo "FAILED"
exit $NG
