# omnivla_real

実機ロボットで [OmniVLA](https://github.com/NHirose/OmniVLA) を動かすための環境です。
構成図: <https://mamoru1126.github.io/omnivla_real/>

- **学習**: 決まったコースを走ったときの rosbag (ROS 1 / ROS 2) をそのまま材料にします。
- **机上評価**: 別の走行の bag を使い、モデルの指示値を積算した軌跡がコースに沿うかを確かめます。
- **走行**: Jetson AGX Orin 上でサブゴール画像を切り替えながら、最終ゴールまで走ります。

| | 入力 | 出力 |
|---|---|---|
| 学習 (RTX 3090 以上) | rosbag: カメラ画像, ホイールオドメトリ, 指示値 (+任意で自己位置) | ファインチューニングした重み |
| 走行（Jetson AGX Orin、ROS 1 / ROS 2） | 今のカメラ画像（+オドメトリ）、サブゴール画像列 | 指示値、予測軌跡 `/omnivla/path`。Humble版の指示値は`/omnivla/cmd_vel/raw` |

モデルは次の 2 つから選べます (学習・走行とも同じ手順)。

- **OmniVLA 7B** (LoRA で学習)
- **OmniVLA-edge** (軽量. Jetson で速い)

## 手順書

| | 内容 |
|---|---|
| [docs/training_pc.md](docs/training_pc.md) | 学習 PC (x86_64 + NVIDIA GPU): 準備、bag の確認、学習、topomap、机上評価、bag での確認 (1〜7)、設定ファイル、テスト |
| [docs/jetson.md](docs/jetson.md) | Jetson AGX Orin: 準備、走行、トピック、デバッグ画面、Jetson での机上評価・bag での確認 (8) |

## 流れ

| | すること | コマンド | どこで |
|---|---|---|---|
| 準備 | `.env` を作り、イメージをビルドして公式の重みを取る | `bin/setup.sh` | 学習 PC / Jetson |
| 1 | bag の中身を見て、`configs/robot.yaml` にトピック名を書く | `bin/bag_info.sh` | 学習 PC |
| 2 | オドメトリが使えるか確かめる | `bin/check_odometry.sh` | 学習 PC |
| 3 | 学習データに変換する | `bin/make_dataset.sh` | 学習 PC |
| 4 | 学習する | `bin/train.sh 7b\|edge` | 学習 PC |
| 5 | サブゴール画像列 (topomap) を作る | `bin/make_topomap.sh` | 学習 PC |
| 6 | 机上評価する | `bin/desk_eval.sh` | 学習 PC (Jetson でも可) |
| 7 | bag をロボットの代わりにして、実機と同じ構成で動かす (ブラウザ / 動画) | `bin/replay.sh` / `bin/replay_video.sh` | 学習 PC (Jetson でも可) |
| 8 | 重みと topomap を送って、Jetson で走らせる | `bin/to_jetson.sh` → `bin/jetson.sh up` | Jetson |

## はじめかた

### JetsonでROS 2 Humbleを使う場合

ROS 1版と同じ推論コンテナを使用します。
ナビゲーションノードだけをHumbleコンテナへ切り替えます。
`.env`に`JETSON_ROS_DISTRO=humble`を設定してください。
`bin/setup.sh edge`と`bin/jetson.sh`はHumble用Compose設定を追加します。
既定の`noetic`では、従来のROS 1版を起動します。

```bash
JETSON_ROS_DISTRO=humble
WITH_7B=0
NAV_MODEL=edge
NAV_WEIGHTS=/checkpoints/omnivla-edge
ROS_DOMAIN_ID=42
CMD_VEL_TOPIC=/omnivla/cmd_vel/raw
DDS_PROFILE_FILE=../nakanoshima_2026/docker/fastdds_usb.xml
```

`JETSON_BASE_IMAGE`には、JetPackに対応したCUDA版PyTorchイメージを指定します。
JetPack 6.2の今回の環境では、既存の
`dustynv/vllm:r36.4.0`でGPUの動作を確認しました。
ホストのCPU版PyTorchは使用しません。

```bash
bin/setup.sh edge
bin/jetson.sh up
bin/jetson.sh status
```

Humble版は`/cmd_vel`を`/omnivla/cmd_vel/raw`へremapします。
ATMOBIの最終出力`/cmd_vel`には直接配信しません。
`--standalone`はROS 1専用のため、Humble版では指定しません。
Humble版のデバッグ情報はROSトピックと走行ログで確認します。
既存のWeb画面はROS 1のC++ノード専用です。

DDS設定は`192.168.1.1`と`192.168.1.100`のUSB接続専用です。
IPアドレスが異なる場合は、DDS設定も変更してください。
ドメイン番号だけでは社内LANへの通信を防げません。
Fast DDSのインターフェース制限とunicast探索を併用します。

USBカメラだけで速度伝達を確認する場合は、次の試験設定を使えます。
試験用topomapには、実際のUSBカメラから保存した画像を配置します。
カメラ画像と同じゴール画像で即座に到達扱いになることを避けるため、
試験設定では到達判定を無効にしています。
実走行用の経路設定ではありません。

```bash
ROBOT_CONFIG=/workspace/configs/robot_usb_test.yaml
NAV_CONFIG=/workspace/configs/navigator_usb_test.yaml
TOPOMAP=/data/topomaps/usb_camera_test
NAV_AUTOSTART=false
```

試験設定はオドメトリを要求しません。
最大速度は`0.05 m/s`、最大角速度は`0.15 rad/s`です。
開始と停止は`/omnivla/enable`の`std_msgs/msg/Bool`で操作します。
推論結果が1秒間更新されない場合は、ゼロ速度を出力します。
実機ではなく、Gazeboへの接続確認に使用してください。

```bash
# 学習 PC
git clone https://github.com/mamoru1126/omnivla_real.git && cd omnivla_real
bin/setup.sh                                 # .env + イメージ + 公式の重み
bin/bag_info.sh run1.bag                     # bags/ に置いた bag を見る → configs/robot.yaml
bin/make_dataset.sh run1.bag run2.bag        # → data/dataset
bin/train.sh 7b                              # → runs/<run>/checkpoints/step_XXXXXX (.env の FINETUNED_DIR に書く)
bin/make_topomap.sh course_a run1.bag        # → data/topomaps/course_a (.env の TOPOMAP に書く)
bin/desk_eval.sh run3.bag course_a           # topomap とは別の走行で評価 → runs/desk_eval/
bin/to_jetson.sh user@jetson                 # Jetson に送る

# Jetson
bin/setup.sh none                            # .env + イメージ (重みは学習 PC から送る)
bin/jetson.sh up                             # 走行. ブラウザで http://<Jetson の IP>:8080
```

- コマンドはどれも `-h` で使い方を表示します。中で `docker compose` を呼ぶので、どこから実行しても構いません。
- モデルと topomap は `.env` の `NAV_MODEL` / `FINETUNED_DIR` / `NAV_WEIGHTS` / `TOPOMAP` を使います (机上評価・bag での確認・Jetson で同じ)。
- 設定ファイルは `configs/` (`robot.yaml`: トピックと画像の前処理、`navigator.yaml`: 走行と机上評価、`convert.yaml`、`finetune_*.yaml`) です。

## 確かめたこと / まだのこと

- 確かめたこと: 実機の bag (ROS 1) での変換・7B の学習・topomap・机上評価、学習 PC での bag を使った確認 (ブラウザ・動画)。CI では合成した bag で全体を通しています ([テスト](docs/training_pc.md#テスト))。
- 2026年10月21日の確認: JetsonのCUDA版PyTorchでedgeモデルを起動しました。実USBカメラをHumbleノードへ入力しました。USB Ethernet限定で`/omnivla/cmd_vel/raw`を本PCへ配信しました。ATMOBIの経由点2で速度入力を切り替え、同じ速度値がGazeboまで届きました。確認後に経由点タスクを正常キャンセルし、Navigationが経由点4へ到着しました。
- まだのこと: Jetsonでの7Bの推論時間、VLAによる実経路の到達性能、実機での走行。

7B の Jetson での推論時間は、Jetson で `bin/desk_eval.sh` を実行すると report.txt の「推論時間」に出ます。遅すぎる場合は edge に切り替えます (学習・走行の手順は同じ)。
