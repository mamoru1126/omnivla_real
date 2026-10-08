#!/usr/bin/env bash
# 合成した rosbag (ROS1 / ROS2 sqlite3 / ROS2 mcap) で、ツールを README の順に通しで動かす (CI 用).
#   bash tests/e2e_cli.sh [OUT_DIR]
# torch が入っていれば OmniVLA-edge の学習 (数 step, CPU) → その重みで机上評価 まで動かす。
set -euo pipefail
OUT=${1:-/tmp/omnivla_e2e}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
rm -rf "$OUT"
mkdir -p "$OUT"
step() { echo; echo "=================== $*"; }

step "make synthetic bags"
python3 tests/bagfiles.py "$OUT/bags" --image_size 160,120 --split            # course_ros1_{0,1}.bag / course_sqlite3 / course_mcap
python3 tests/bagfiles.py "$OUT/bags_lap2" --image_size 160,120 --formats mcap --speed 0.5 --slip 0.95 --no_stop
ROS1="$OUT/bags/course_ros1_0.bag,$OUT/bags/course_ros1_1.bag"
SQL="$OUT/bags/course_sqlite3"
MCAP="$OUT/bags/course_mcap"
EVAL="$OUT/bags_lap2/course_mcap"

# 自己位置あり / なし の robot.yaml
sed 's#^  localization: ""#  localization: /localization#' configs/robot.yaml > "$OUT/robot_loc.yaml"
grep -q "localization: /localization" "$OUT/robot_loc.yaml"

step "bag_info"
for b in "$ROS1" "$SQL" "$MCAP"; do python3 tools/bag_info.py "$b"; done

step "check_odometry"
python3 tools/check_odometry.py --robot "$OUT/robot_loc.yaml" --out "$OUT/check_odometry" "$MCAP"
ls "$OUT/check_odometry"

step "bag_to_dataset (ROS1 split + ROS2 sqlite3 + ROS2 mcap)"
python3 tools/bag_to_dataset.py --robot configs/robot.yaml --config configs/convert.yaml --out "$OUT/dataset" \
    "$ROS1" "$SQL"
python3 tools/bag_to_dataset.py --robot configs/robot.yaml --config configs/convert.yaml --out "$OUT/dataset" \
    --name lap2 "$EVAL"
python3 tools/bag_to_dataset.py --robot "$OUT/robot_loc.yaml" --config configs/convert.yaml \
    --out "$OUT/dataset_loc" --pose_source localization "$MCAP"
cat "$OUT/dataset/dataset_info.json"

step "inspect_dataset"
python3 training/inspect_dataset.py "$OUT/dataset" --num_viz 4 --out "$OUT/inspect"

step "make_topomap (odom / localization)"
python3 tools/make_topomap.py --robot configs/robot.yaml --out "$OUT/topomap" --spacing 1.0 "$MCAP"
python3 tools/make_topomap.py --robot "$OUT/robot_loc.yaml" --out "$OUT/topomap_loc" --spacing 1.0 "$MCAP"

step "desk_eval (oracle = 記録の正解軌跡を返す. 評価の仕組みの確認)"
python3 tools/desk_eval.py --robot configs/robot.yaml --bag "$EVAL" --topomap "$OUT/topomap" \
    --policy oracle --out "$OUT/desk_eval_oracle"
python3 tools/desk_eval.py --robot "$OUT/robot_loc.yaml" --bag "$EVAL" --topomap "$OUT/topomap_loc" \
    --policy oracle --out "$OUT/desk_eval_oracle_loc"
python3 - "$OUT" <<'EOF'
import json, sys
for d, mode in (("desk_eval_oracle", "image_odom"), ("desk_eval_oracle_loc", "pose")):
    r = json.load(open(f"{sys.argv[1]}/{d}/report.json"))
    sg = r["subgoals"]
    print(d, "reach_check", sg.get("reach_check"), "reached", sg.get("reached_goal"),
          "turn_agree", r["commands"].get("turn_direction_agreement"))
    assert sg.get("reached_goal") and sg.get("reach_check") == mode, sg
    assert r["commands"]["turn_direction_agreement"] > 0.9
EOF

if python3 -c "import torch, efficientnet_pytorch" 2>/dev/null; then
    step "finetune_edge (from scratch, CPU, a few steps)"
    python3 training/finetune_edge.py --config configs/finetune_edge.yaml --weights "" --clip_type "" \
        --device cpu --data_dirs "$OUT/dataset" --val_bags lap2 --max_steps 4 --batch_size 4 --num_workers 0 \
        --val_freq 2 --val_batches 2 --num_viz 2 --save_freq 4 --lr_warmup_steps 1 --amp false \
        --run_root "$OUT/runs" --run_name edge_ci
    CKPT="$OUT/runs/edge_ci/checkpoints/step_000004"
    ls "$CKPT"

    step "desk_eval (OmniVLA-edge, CPU)"
    python3 tools/desk_eval.py --robot configs/robot.yaml --bag "$EVAL" --topomap "$OUT/topomap" \
        --model edge --weights "$CKPT" --device cpu --end_sec 30 --debug_every 20 --out "$OUT/desk_eval_edge"
    head -40 "$OUT/desk_eval_edge/report.txt"

    step "policy_server + desk_eval --model remote (ROS 1 コンテナから推論サーバを呼ぶ構成) == プロセス内推論"
    python3 tools/policy_server.py --model edge --weights "$CKPT" --device cpu --port 8799 > "$OUT/policy_server.log" 2>&1 &
    SERVER=$!
    trap 'kill $SERVER 2>/dev/null || true' EXIT
    python3 tools/desk_eval.py --robot configs/robot.yaml --bag "$EVAL" --topomap "$OUT/topomap" \
        --model remote --url http://127.0.0.1:8799 --end_sec 30 --out "$OUT/desk_eval_remote"
    kill $SERVER
    cat "$OUT/policy_server.log"
    python3 - "$OUT" <<'PYEOF'
import json, math, sys
a = json.load(open(f"{sys.argv[1]}/desk_eval_edge/report.json"))
b = json.load(open(f"{sys.argv[1]}/desk_eval_remote/report.json"))
n = 0
for sec in ("waypoints", "commands"):
    for k, v in a[sec].items():
        w = b[sec].get(k)
        if isinstance(v, float) and not math.isnan(v):
            assert abs(v - w) < 1e-4, (sec, k, v, w)
            n += 1
assert n > 3, a
print(f"remote == local ({n} metrics)")
PYEOF
else
    echo "(torch / efficientnet_pytorch not installed: skip edge training and model desk_eval)"
fi
echo; echo "e2e OK -> $OUT"
