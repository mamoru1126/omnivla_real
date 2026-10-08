# omnivla_real

実機ロボットで [OmniVLA](https://github.com/NHirose/OmniVLA) を動かすための環境です。

- **学習**: 決まったコースを走ったときの rosbag (ROS 1 / ROS 2) をそのまま材料にします。
- **机上評価**: 別の走行の bag を使い、モデルの指示値を積算した軌跡がコースに沿うかを確かめます。
- **走行**: Jetson AGX Orin 上でサブゴール画像を切り替えながら、最終ゴールまで走ります。

| | 入力 | 出力 |
|---|---|---|
| 学習 (RTX 3090 以上) | rosbag: カメラ画像, ホイールオドメトリ, 指示値 (+任意で自己位置) | ファインチューニングした重み |
| 走行 (Jetson AGX Orin) | 今のカメラ画像 (+オドメトリ), サブゴール画像列 | 指示値 `/cmd_vel`, 予測軌跡 (約 2.7 秒 / 数 m) `/omnivla/path` |

モデルは次の 2 つから選べます。

- **OmniVLA-edge** (軽量, Jetson 向け, 既定)
- **OmniVLA 7B** (LoRA で学習)

## 流れ

1. bag の中身を見て、`configs/robot.yaml` にトピック名を書く (`bag_info.py`)
2. オドメトリが使えるか確かめる (`check_odometry.py`)
3. 学習データに変換する (`bag_to_dataset.py`)
4. 学習する (`finetune_edge.py` / `finetune_omnivla.py`)
5. サブゴール画像列 (topomap) を作る (`make_topomap.py`)
6. 机上評価する (`desk_eval.py`)
7. Jetson で走らせる (ROS 2 / ROS 1 ノード)

## 準備 (学習 PC: x86_64 + NVIDIA GPU)

Docker と NVIDIA Container Toolkit が必要です。

```bash
git clone https://github.com/mamoru1126/omnivla_real.git && cd omnivla_real
cp .env.example .env
docker compose build
docker compose run --rm shell bash scripts/download_checkpoints.sh edge   # 7B も使うなら引数なし
```

フォルダとコンテナ内のパスの対応 (`.env` で変更可):

| ホスト | コンテナ |
|---|---|
| `./bags` | `/bags` |
| `./data` | `/data` (学習データ・topomap) |
| `./runs` | `/runs` (学習・評価の結果) |
| `./checkpoints` | `/checkpoints` (公式の重み) |

以下のコマンドは `docker compose run --rm shell` の中で実行します。

## 1. bag の中身を見る

```bash
python3 tools/bag_info.py /bags/run1          # ROS 2: bag のディレクトリ (.db3 / .mcap)
python3 tools/bag_info.py /bags/run1.bag      # ROS 1
```

表示されたトピック名を `configs/robot.yaml` の `topics` に書きます。

| 種類 | 対応している型 |
|---|---|
| 画像 | `Image` / `CompressedImage` |
| ホイールオドメトリ | `nav_msgs/Odometry` |
| 指示値 | `Twist` / `TwistStamped` / 独自型 (フィールドを指定) |
| 自己位置 (任意) | `PoseStamped` / `PoseWithCovarianceStamped` / `Odometry` / `/tf` |

- 車体が画像に写り込む場合は `image.crop` で削ります。
- 分割された bag は `a_0.bag,a_1.bag` のようにカンマでつなぐと、1 本の走行として扱います (以降のツールも同じ)。

## 2. オドメトリが使えるか確かめる

```bash
python3 tools/check_odometry.py --out /runs/check/run1 /bags/run1
```

学習ラベルは「この先約 2.7 秒の動き」です。ラベルに使える位置の候補は 3 つ (自己位置があれば 4 つ) あり、このツールはそれらの短時間の動きがどれだけ一致するかを比べます。

- ホイールオドメトリ
- オドメトリの速度の積算
- 指示値の積算

あわせて、指示値から実際に動き出すまでの遅れと、速度の比 (スリップ) も推定します。

- odom と cmd の差が数 cm なら、既定の `pose_source: odom` で OK です。
- オドメトリが信用できない場合は `pose_source: cmd` (+ `cmd_delay`) にします。
- 自己位置がある場合は `pose_source: localization` も選べます。

## 3. 学習データに変換する

```bash
python3 tools/bag_to_dataset.py --out /data/dataset /bags/run1 /bags/run2.bag
python3 training/inspect_dataset.py /data/dataset --num_viz 16 --out /runs/inspect   # 正解軌跡を画像に重ねて確認
```

- bag はエピソード単位でなくて構いません。変換時に次のように区切ります。
  - 3 Hz で間引く。
  - 止まっている時間 (3 秒超)、画像やオドメトリの途切れ、後退で区切る。
  - 長い区間は 60 秒ごとに分ける。
- 走り出す前の停止や途中の停止は、学習の起点から外します。区間の最後の停止は「ゴールで止まる」学習に使います。
- どこを使ったかは `/data/dataset/_reports/<bag>.png` で確認できます。設定は `configs/convert.yaml` にあります。
- 同じコースでも、時間帯・明るさ・走る位置 (少し左右にずれる) を変えて何回か走った bag を入れると強くなります。
- コースから外れて戻る走りも入れると、外れたときに復帰できるようになります。シミュレータの DART のような自動の外乱は、実データでは使えません。

## 4. 学習

```bash
docker compose run --rm train_edge     # OmniVLA-edge (数 GB の GPU で可). configs/finetune_edge.yaml
docker compose run --rm train_7b       # OmniVLA 7B, LoRA (24GB 以上). configs/finetune_7b.yaml
```

- 結果は `/runs/<run>/checkpoints/step_XXXXXX/` に出ます。
- 検証に使う走行は `val_bags: [run2]` のように bag 名で指定できます。
- 7B をマージ済みモデルにしたい場合は `training/merge_lora.py` を使います。

## 5. サブゴール画像列 (topomap) を作る

コースを 1 回走った bag から、1 m ごとに画像を切り出します。

```bash
python3 tools/make_topomap.py --out /data/topomaps/course_a --spacing 1.0 /bags/run1
```

- 出力は `0.jpg, 1.jpg, ...`、`poses.yaml` (各画像の位置)、`overview.png` です。
- 使う範囲は `--start_sec` / `--end_sec` で切り出せます。

## 6. 机上評価

topomap を作った走行とは別の走行の bag で評価します。録画した画像を順にモデルへ入れ、走行時と同じ処理をして、記録と比べます。

- サブゴールの切り替え
- 推論
- 予測軌跡 → 指示値

```bash
python3 tools/desk_eval.py --bag /bags/run2 --topomap /data/topomaps/course_a \
    --model edge --weights /runs/<run>/checkpoints/step_010000 --out /runs/desk_eval/run2 --debug_every 10
```

| 出力 | 中身 |
|---|---|
| `overview.png` | 実際の走行、**モデルの指示値を積算した軌跡**、記録の指示値を積算した軌跡、サブゴール |
| `timeline.png` | 速度・角速度 (モデル vs 記録)、サブゴール番号、画像の類似度 |
| `report.txt` / `report.json` | 下の表の指標 |
| `steps.csv` | フレームごとの値 |
| `debug/` | 予測軌跡を画像に重ねたもの |

| 指標 | 意味 |
|---|---|
| `waypoints.ade_m` | 予測した 8 点と、実際にその後走った 8 点のずれ |
| `commands.turn_direction_agreement` | 曲がる場面で、モデルの旋回の向きが記録と合っている割合 |
| `integrated_windows` | 実際の位置から N 秒間、モデルの指示値だけで進めたときの終点のずれ。記録の指示値で同じことをした値 (`recorded_cmd`) と同程度なら十分 |
| `subgoals.reached_goal` | 最後のサブゴールまで切り替わったか。`reach_check` は判定の方法 |

- 入力は録画なので開ループです (モデルが曲がり損ねても、画像は記録どおりに進みます)。スタートから積算した軌跡 (`integrated_full`) は誤差が溜まるので、主に N 秒窓で見ます。
- `--policy oracle` を付けると、モデルの代わりに記録の正解を返します。評価やサブゴール切り替えの仕組みの確認に使います。
- `--goal_mode hindsight` にすると、topomap を使わず「記録の 1.5 m 先の画像」をゴールにします。

サブゴールの切り替え方 (`configs/navigator.yaml` の `engine.tracker.reach_check: auto`):

| 状況 | 判定 |
|---|---|
| 自己位置がある | 位置で判定 (`pose`) |
| 自己位置がない | 画像の類似度 + オドメトリで判定 (`image_odom`)。オドメトリで遠すぎるサブゴールは候補にしない。画像で判定できないまま通り過ぎたら進める |

画像の類似度のしきい値 (`image_threshold`) は、机上評価の `report.json` の `similarity_threshold` を見て決めます。

## 7. Jetson AGX Orin で走らせる

JetPack 6 (L4T r36) を前提とし、ROS 2 Humble 入りのイメージを使います。学習 PC から重みと topomap をコピーしておきます。

```bash
git clone https://github.com/mamoru1126/omnivla_real.git && cd omnivla_real
cp .env.example .env              # JetPack が違う場合は docker/Dockerfile.jetson の BASE_IMAGE を合わせる
docker compose -f docker-compose.jetson.yml build
# ./runs/<run>/checkpoints/step_010000 と ./data/topomaps/course_a を置いてから
TOPOMAP=/data/topomaps/course_a FINETUNED_DIR=/runs/<run>/checkpoints/step_010000 \
    docker compose -f docker-compose.jetson.yml run --rm nav
```

カメラ画像とオドメトリ (自己位置があればそれも) が届くと、走り出します (`autostart`)。

| | トピック | 型 |
|---|---|---|
| 入力 | `robot.yaml` の `topics.image` / `odom` / `localization` | |
| 入力 | `/omnivla/enable` | `std_msgs/Bool`。開始 / 停止 |
| 入力 | `/omnivla/topomap` | `std_msgs/String`。topomap を切り替えて開始 |
| 出力 | `/cmd_vel` | `Twist` (`io.cmd_stamped: true` で `TwistStamped`) |
| 出力 | `/omnivla/path` | `nav_msgs/Path`。予測軌跡 8 点 (`base_link` 座標) |
| 出力 | `/omnivla/debug_image` | 予測軌跡を重ねた画像 (rqt_image_view で見る) |
| 出力 | `/omnivla/status` | JSON。状態, サブゴール番号, 類似度, 推論時間 |

- 速度の上限は `navigator.yaml` の `engine.controller.track_max_v` / `track_max_w` で、ロボットに合わせます。
- 推論結果が `io.cmd_timeout` 秒より古くなると 0 を出します。
- 走行ログは `log/nav/<時刻>/` に保存されます。
  - `python3 tools/plot_nav_log.py log/nav/latest` で、図とレポートを作れます。
- bag の再生でも試せます。
  ```bash
  ros2 launch omnivla_real_ros navigator.launch.py topomap:=... use_sim_time:=true
  ros2 bag play <bag> --clock --topics <画像> <odom>
  ```

ROS 1 の場合は `roslaunch omnivla_real_ros1 navigator.launch topomap:=...` で起動します。中身は ROS 2 版と共通です。rospy と PyTorch が同じ Python で動く環境が必要です。JetPack 6 (Ubuntu 22.04) には ROS 1 が無いので、Jetson では ROS 2 版 + ros1_bridge を推奨します。

## 設定ファイル

| ファイル | 内容 |
|---|---|
| `configs/robot.yaml` | トピック名、画像の前処理 (変換・topomap・評価・走行で共通)、カメラの値 (表示用) |
| `configs/convert.yaml` | bag → 学習データの変換 |
| `configs/finetune_edge.yaml` / `finetune_7b.yaml` | 学習 |
| `configs/navigator.yaml` | 走行と机上評価: モデル、制御、サブゴールの切り替え、トピック |

## テスト

```bash
python3 -m pytest -q tests                 # 単体テスト + 本物の rosbag を書いて読む
bash tests/e2e_cli.sh /tmp/e2e             # 合成した bag で 1〜6 を通しで実行
```

CI (GitHub Actions) では合成したコース走行の bag を使い、次を確認しています。

- ROS 1 / ROS 2 sqlite3 / ROS 2 mcap の bag を読む
- 変換 → edge の学習 (CPU で数 step) → topomap → 机上評価
- ROS 2 / ROS 1 ノードに bag を再生して、`/cmd_vel` と `/omnivla/path` が出る

## 未確認のこと

- 実機の bag、実機での走行
- Jetson 用イメージのビルド
- 7B の GPU での学習・推論

これらはまだ試していません。7B は Jetson では遅い可能性が高いので、まず edge を推奨します。
