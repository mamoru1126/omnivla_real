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
export OMNIVLA_DRY_RUN=1 BAG_DIR=$T/bags DATA_DIR=$T/data RUNS_DIR=$T/runs CHECKPOINT_DIR=$T/checkpoints
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

# --- 学習 PC で実機と同じ構成 ---
check "up -d --force-recreate --no-deps nav record robot" "$B/replay.sh" run1.bag course_a --rate 2 --manual
check "bag: /bags/split/a_0.bag,/bags/split/a_1.bag  topomap: /data/topomaps/course_a" "$B/replay.sh" split/a_0.bag,split/a_1.bag course_a
check "docker compose -f docker-compose.replay.yml logs -f --tail 50 robot nav" "$B/replay.sh" run1.bag course_a
check "docker compose -f docker-compose.replay.yml restart nav" "$B/replay.sh" again
check "docker compose -f docker-compose.replay.yml down" "$B/replay.sh" down
fails "topomap を指定" "$B/replay.sh" run1.bag
fails "知らないオプション" "$B/replay.sh" run1.bag course_a --fast
check "tools/plot_nav_log.py /runs/replay/nav/latest" "$B/plot_nav_log.sh"

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
check "docker compose -f docker-compose.jetson.yml --profile standalone down" "$B/jetson.sh" down

# --- 使い方の表示 ---
for f in "$B"/*.sh; do
    [ "$(basename "$f")" = _lib.sh ] && continue
    check "bin/$(basename "$f")" "$f" -h
done

[ $NG = 0 ] && echo "all OK" || echo "FAILED"
exit $NG
