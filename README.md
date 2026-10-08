# omnivla_real

実機ロボットで [OmniVLA](https://github.com/NHirose/OmniVLA) を動かすための環境です。
構成図: <https://mamoru1126.github.io/omnivla_real/>

- **学習**: 決まったコースを走ったときの rosbag (ROS 1 / ROS 2) をそのまま材料にします。
- **机上評価**: 別の走行の bag を使い、モデルの指示値を積算した軌跡がコースに沿うかを確かめます。
- **走行**: Jetson AGX Orin 上でサブゴール画像を切り替えながら、最終ゴールまで走ります。

| | 入力 | 出力 |
|---|---|---|
| 学習 (RTX 3090 以上) | rosbag: カメラ画像, ホイールオドメトリ, 指示値 (+任意で自己位置) | ファインチューニングした重み |
| 走行 (Jetson AGX Orin, ROS 1) | 今のカメラ画像 (+オドメトリ), サブゴール画像列 | 指示値 `/cmd_vel`, 予測軌跡 (約 2.7 秒 / 数 m) `/omnivla/path` |

モデルは次の 2 つから選べます (学習・走行とも同じ手順)。

- **OmniVLA 7B** (LoRA で学習)
- **OmniVLA-edge** (軽量. Jetson で速い)

## 流れ

1. bag の中身を見て、`configs/robot.yaml` にトピック名を書く (`bag_info.py`)
2. オドメトリが使えるか確かめる (`check_odometry.py`)
3. 学習データに変換する (`bag_to_dataset.py`)
4. 学習する (`finetune_edge.py` / `finetune_omnivla.py`)
5. サブゴール画像列 (topomap) を作る (`make_topomap.py`)
6. 机上評価する (`desk_eval.py`)
7. bag を再生して ROS 1 ノードを試す (PC)
8. Jetson で走らせる (ROS 1)

## 準備 (学習 PC: x86_64 + NVIDIA GPU)

Docker と NVIDIA Container Toolkit が必要です。

```bash
git clone https://github.com/mamoru1126/omnivla_real.git && cd omnivla_real
cp .env.example .env
docker compose build
docker compose run --rm shell bash scripts/download_checkpoints.sh        # 公式の重み (7B と edge). edge だけなら引数 edge
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
python3 tools/bag_info.py /bags/run1.bag      # ROS 1
python3 tools/bag_info.py /bags/run1          # ROS 2: bag のディレクトリ (.db3 / .mcap)
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
docker compose run --rm train_7b       # OmniVLA 7B, LoRA (24GB 以上). configs/finetune_7b.yaml
docker compose run --rm train_edge     # OmniVLA-edge (数 GB の GPU で可). configs/finetune_edge.yaml
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
python3 tools/desk_eval.py --bag /bags/run2.bag --topomap /data/topomaps/course_a \
    --model 7b --finetuned_dir /runs/<run>/checkpoints/step_005000 --out /runs/desk_eval/run2 --debug_every 10
# edge: --model edge --weights /runs/<run>/checkpoints/step_010000
# 推論サーバ経由 (実機と同じ): --model remote --url http://127.0.0.1:8765
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

## 7. bag を再生して ROS 1 ノードを試す (PC)

実機と同じ構成 (推論サーバ + ROS 1 ノード) で、別の走行の bag を流して動きを確認します。

```bash
# .env に NAV_MODEL / FINETUNED_DIR を書いておく
docker compose up -d policy                       # 推論サーバ (GPU)
docker compose run --rm ros1 bash scripts/replay_bag_ros1.sh /bags/run2.bag /data/topomaps/course_a
```

- bag からは `robot.yaml` の画像・オドメトリ (・自己位置) だけを流します。記録された `/cmd_vel` は流しません。
- 出力は `/runs/replay/<時刻>/` に出ます。
  - `out.bag`: ノードが出した `/cmd_vel` と `/omnivla/path`
  - `nav/`: 走行ログと `overview.png`
- 再生中はブラウザで http://localhost:8080 を開くと、デバッグ画面 (下の「デバッグ画面」) で様子を見られます。

## 8. Jetson AGX Orin で走らせる (ROS 1)

コンテナを 2 つに分け、localhost の HTTP でつなぎます。

| コンテナ | 中身 | Dockerfile |
|---|---|---|
| `policy` | 推論サーバ (Python)。OmniVLA を GPU で動かす。ROS なし | `docker/Dockerfile.jetson` |
| `nav` | ROS 1 Noetic のナビゲーションノード (C++) とデバッグ画面。PyTorch なし | `docker/Dockerfile.ros1` |

分けている理由: ROS 1 Noetic は Ubuntu 20.04 用ですが、Jetson で GPU を使える PyTorch は JetPack ごとに Ubuntu が決まっています (JetPack 6 は 22.04)。分けておけば、JetPack が決まっていなくても ROS 1 側はそのまま使えます。

推論サーバのベースイメージは、JetPack に合わせて `.env` の `JETSON_BASE_IMAGE` で選びます。

| JetPack | `JETSON_BASE_IMAGE` |
|---|---|
| 6.1 / 6.2 (L4T r36.4) | `dustynv/l4t-pytorch:r36.4.0` (既定) |
| 6.0 (L4T r36.2) | `dustynv/l4t-pytorch:r36.2.0` |
| 5.1.2 (L4T r35.4) | `dustynv/l4t-pytorch:2.2-r35.4.1` |

```bash
git clone https://github.com/mamoru1126/omnivla_real.git && cd omnivla_real
cp .env.example .env    # JETSON_BASE_IMAGE, NAV_MODEL, FINETUNED_DIR, TOPOMAP, ROS_MASTER_URI を書く
docker compose -f docker-compose.jetson.yml build
# 学習 PC から ./runs/<run>/checkpoints/step_XXXXXX と ./data/topomaps/course_a をコピー
# (7B の場合は ./checkpoints/omnivla-original も)
docker compose -f docker-compose.jetson.yml up               # ROS master は ROS_MASTER_URI のもの
docker compose -f docker-compose.jetson.yml --profile standalone up   # roscore もここで立てる場合
```

- `nav` は推論サーバの準備ができるのを待ってから始まります。7B の読み込みには数分かかります。
- カメラ画像とオドメトリ (自己位置があればそれも) が届くと走り出します (`autostart`)。

| | トピック | 型 |
|---|---|---|
| 入力 | `robot.yaml` の `topics.image` / `odom` / `localization` | |
| 入力 | `/omnivla/enable` | `std_msgs/Bool`。開始 / 停止 |
| 入力 | `/omnivla/topomap` | `std_msgs/String`。topomap を切り替えて開始 |
| 出力 | `/cmd_vel` | `Twist` (`io.cmd_stamped: true` で `TwistStamped`) |
| 出力 | `/omnivla/path` | `nav_msgs/Path`。予測軌跡 8 点 (`base_link` 座標) |
| 出力 | `/omnivla/status` | JSON。状態, サブゴール番号, 類似度, 推論時間 |
| ブラウザ | http://&lt;Jetson の IP&gt;:8080 | デバッグ画面 |

- 速度の上限は `navigator.yaml` の `engine.controller.track_max_v` / `track_max_w` で、ロボットに合わせます。
- 推論結果が `io.cmd_timeout` 秒より古くなると 0 を出します。推論サーバが止まった場合も 0 になります。
- 走行ログは `log/nav/<時刻>/` に保存されます。
  - `python3 tools/plot_nav_log.py log/nav/latest` で、図とレポートを作れます。
- ROS 1 ノードは既存の ROS 1 環境 (catkin ワークスペース) に入れても動きます。
  - `ros1/omnivla_real_ros1` を置いて `catkin_make` します。C++ の依存は同梱のヘッダだけです。
  - 起動時に設定を読むため、python3 と PyYAML / numpy / Pillow が要ります。`OMNIVLA_REAL_ROOT` に本リポジトリを指定します。
- 推論サーバは別の PC (学習 PC など) で動かしても構いません。その場合はサーバを `--host 0.0.0.0` で起動し、`policy_url:=http://<IP>:8765` を指定します。
- Python 版の ROS 1 ノードも残しています (`roslaunch ... impl:=py`)。デバッグ画面は C++ 版だけです。

ROS 2 で使う場合は `ros2 launch omnivla_real_ros navigator.launch.py` で起動します (ROS 2 ノードも同梱)。

### デバッグ画面

走行中にブラウザで http://&lt;Jetson の IP&gt;:8080 を開きます (ロボットと同じネットワークの PC やタブレットから)。

| 表示 | 中身 |
|---|---|
| いまの画像 | モデルに入れた画像 (切り抜き・縮小の後) に、予測した 8 点の軌跡を重ねたもの |
| サブゴール | いま目指しているサブゴール画像と、そこまでの距離。下にコースのサブゴールが並び、今の位置が分かる |
| 指示値 | ロボットに出している v と ω、直近 60 秒の履歴 |
| 予測した軌跡 | 上から見た図 (0.5 m 方眼) |
| サブゴールの判定 | 画像の類似度と切り替えのしきい値、推論時間と回数 |
| 出来事 | サブゴールの切り替え、開始・停止 |

- 画面の「停止」ボタンか Esc キーで止まります。「開始」は確認してから動き出します。
- 画面を開いている人がいないときは、画面用の画像を作りません (走行の負荷は増えません)。
- 認証はありません。ロボットの中のネットワークだけで使い、外から見せたくないときは `navigator.yaml` の `io.web_host: 127.0.0.1` にします。ポートは `io.web_port` (0 で無効) です。
- 画面を閉じたり接続が切れたりしても、ノードは走り続けます。止めるときは停止ボタンかロボット側で止めます。

### 軽くするためにしていること

- ROS 1 ノードは C++ です。カメラの JPEG はデコードせずに、そのまま推論サーバとブラウザに渡します (メッセージのコピーもしません)。
- 画像の切り抜き・縮小は推論サーバ側で、学習時と同じ関数で行います。
- 推論サーバは送られた画像を覚えています。観測履歴とサブゴール画像は 2 回目から名前だけを送ります。
- 推論は新しいカメラ画像が来たときだけです。指示値は 10Hz で最新の結果を出し続けます。
- PyTorch で動かすモデル本体 (推論サーバ) は Python のままです。

## 設定ファイル

| ファイル | 内容 |
|---|---|
| `configs/robot.yaml` | トピック名、画像の前処理 (変換・topomap・評価・走行で共通)、カメラの値 (表示用) |
| `configs/convert.yaml` | bag → 学習データの変換 |
| `configs/finetune_edge.yaml` / `finetune_7b.yaml` | 学習 |
| `configs/navigator.yaml` | 走行と机上評価: モデル、制御、サブゴールの切り替え、トピック |

## テスト

```bash
python3 -m pytest -q tests                 # 単体テスト + 本物の rosbag を書いて読む + C++ と Python の突き合わせ
bash tests/e2e_cli.sh /tmp/e2e             # 合成した bag で 1〜6 を通しで実行
```

CI (GitHub Actions) では合成したコース走行の bag を使い、次を確認しています。

- ROS 1 / ROS 2 sqlite3 / ROS 2 mcap の bag を読む
- 変換 → edge の学習 (CPU で数 step) → topomap → 机上評価
- 推論サーバ経由の推論が、プロセス内の推論と同じ結果になる
- C++ のノードの中身 (制御・サブゴールの切り替え・走行全体) が Python の実装と同じ結果になる (`tests/test_cpp.py`)
- ROS 1: 推論サーバ + ROS 1 コンテナ (`docker/Dockerfile.ros1`) で bag を再生し、`/cmd_vel` と `/omnivla/path` が出る。デバッグ画面に結果と画像が届く (C++ / Python 両方のノード、`scripts/replay_bag_ros1.sh` も)
- ROS 2 ノードも同様
- 学習用イメージのビルド、Jetson 用推論サーバの Dockerfile (x86 の PyTorch イメージを代わりのベースにして依存のインストールまで)

## 未確認のこと

- 実機の bag、実機での走行
- Jetson 上でのイメージのビルド (推論サーバ・ROS 1 とも)
- 7B の GPU での学習・推論

これらはまだ試していません。7B の Jetson での推論時間は未計測です。遅すぎる場合は edge に切り替えます (学習・走行の手順は同じ)。
