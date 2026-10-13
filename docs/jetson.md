# Jetson AGX Orin での手順 (ROS 1)

[README](../README.md) の 8 です。学習・topomap・机上評価 (1〜7) は [training_pc.md](training_pc.md) にあります。

## コマンド (`bin/`)

| コマンド | どこで | 中でしていること |
|---|---|---|
| `bin/to_jetson.sh user@jetson [--bag BAG] [--configs]` | 学習 PC | 重み・topomap (・評価用の bag・設定) を rsync で送る |
| `bin/jetson.sh build\|up [--standalone]\|restart\|status\|logs\|plot\|down` | Jetson | `docker-compose.jetson.yml` (推論サーバ + ROS 1 ノード) |
| `bin/desk_eval.sh` / `replay.sh` / `replay_video.sh` / `plot_nav_log.sh` など | Jetson | 学習 PC と同じ確認 ([下](#jetson-で机上評価bag-での確認)) |

- 使い方は `-h` で表示します。`OMNIVLA_DRY_RUN=1` を付けると、実行せずに中のコマンドを表示します。

## 構成

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

## 準備と走行

```bash
# Jetson で
git clone https://github.com/mamoru1126/omnivla_real.git && cd omnivla_real
cp .env.example .env    # JETSON_BASE_IMAGE, ROS_MASTER_URI を書く
bin/jetson.sh build

# 学習 PC で: .env の NAV_MODEL / FINETUNED_DIR / TOPOMAP の重みと topomap を送る (7B は元のモデルも)
bin/to_jetson.sh user@jetson          # 最後に Jetson の .env に書く行が表示される

# Jetson で
bin/jetson.sh up                      # ROS master は ROS_MASTER_URI のもの. ログを表示 (Ctrl-C で表示だけ止まる)
bin/jetson.sh up --standalone         # roscore もここで立てる場合
bin/jetson.sh status                  # 状態と推論サーバの応答
bin/jetson.sh down                    # 止める
```

- `nav` は推論サーバの準備ができるのを待ってから始まります。7B の読み込みには数分かかります。
- カメラ画像とオドメトリ (自己位置があればそれも) が届くと走り出します (`autostart`)。`.env` の `NAV_AUTOSTART=false` にすると、ブラウザの「開始」か `/omnivla/enable` で走り出します。

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
  - `bin/jetson.sh plot` で、一番新しい走行の図とレポートを作れます。学習 PC に持ってきたものは `bin/plot_nav_log.sh log/nav/<時刻>`。
- ROS 1 ノードは既存の ROS 1 環境 (catkin ワークスペース) に入れても動きます。
  - `ros1/omnivla_real_ros1` を置いて `catkin_make` します。C++ の依存は同梱のヘッダだけです。
  - 起動時に設定を読むため、python3 と PyYAML / numpy / Pillow が要ります。`OMNIVLA_REAL_ROOT` に本リポジトリを指定します。
- 推論サーバは別の PC (学習 PC など) で動かしても構いません。その場合はサーバを `--host 0.0.0.0` で起動し、`policy_url:=http://<IP>:8765` を指定します。
- Python 版の ROS 1 ノードも残しています (`roslaunch ... impl:=py`)。デバッグ画面は C++ 版だけです。

ROS 2 で使う場合は `ros2 launch omnivla_real_ros navigator.launch.py` で起動します (ROS 2 ノードも同梱)。

## Jetson で机上評価・bag での確認

学習 PC で行った確認 ([training_pc.md](training_pc.md) の 6, 7) は、Jetson の上でも同じコマンドで動きます。モデルは実機の走行と同じ推論サーバ (`docker-compose.jetson.yml` の `policy`) を使うので、**Jetson での推論時間**も含めて確かめられます。

```bash
# 学習 PC で: 評価に使う bag も送る
bin/to_jetson.sh user@jetson --bag 20260919_150554.bag

# Jetson で
bin/desk_eval.sh 20260919_150554.bag course_a      # 机上評価. report.txt の「推論時間」が Jetson での値
bin/replay.sh 20260919_150554.bag course_a         # bag をロボットの代わりにして、ブラウザで確認
bin/replay_video.sh 20260919_150554.bag course_a   # 同じ構成で動画に
bin/replay.sh down && bin/jetson.sh up             # 実機の走行に戻す
```

| コマンド | 学習 PC | Jetson |
|---|---|---|
| `bin/desk_eval.sh` | 学習用コンテナでモデルを直接読み込む | 推論サーバを起動して使う (`--model remote`)。動いていればそのまま |
| `bin/replay.sh` / `replay_video.sh` | 推論サーバは `docker-compose.yml` の `policy` | 推論サーバは `docker-compose.jetson.yml` の `policy` (`REPLAY_POLICY_COMPOSE`) |
| `bin/bag_info.sh` / `check_odometry.sh` / `make_topomap.sh` / `plot_nav_log.sh` / `shell.sh` | 学習用コンテナ | 推論サーバのイメージ (bag は rosbags で読む) |
| `bin/train.sh` / `make_dataset.sh` | ○ | 学習 PC で |

- Jetson かどうかは `/etc/nv_tegra_release` で見分けます。`OMNIVLA_JETSON=1` / `0` で決め打ちもできます。
- bag を流している間、実機の走行用の `nav` は bag 用に置き換わります。ROS master は Jetson の中だけですが、念のためロボットのモーターは切っておきます。
- bag を読むもの (rosbags) は Python 3.10 以上が要るので JetPack 6 だけです。JetPack 5 では学習 PC で行います。
- 机上評価と bag での確認を推論サーバ経由にしているのは、モデルを 2 つ読み込まない (7B は GPU のメモリを多く使う) ためと、走行と同じ通信込みの時間を測るためです。

## デバッグ画面

走行中にブラウザで http://&lt;Jetson の IP&gt;:8080 を開きます (ロボットと同じネットワークの PC やタブレットから)。

![デバッグ画面](debug_page.jpg)

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

## 軽くするためにしていること

- ROS 1 ノードは C++ です。カメラの JPEG はデコードせずに、そのまま推論サーバとブラウザに渡します (メッセージのコピーもしません)。
- 画像の切り抜き・縮小は推論サーバ側で、学習時と同じ関数で行います。
- 推論サーバは送られた画像を覚えています。観測履歴とサブゴール画像は 2 回目から名前だけを送ります。
- 推論は新しいカメラ画像が来たときだけです。指示値は 10Hz で最新の結果を出し続けます。
- PyTorch で動かすモデル本体 (推論サーバ) は Python のままです。
