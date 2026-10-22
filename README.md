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
DDS_PROFILE_FILE=../nakanoshima_2026/docker/fastdds_robot_links.xml
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

DDSの既定設定は、専用LANと検証用USB Ethernetの両経路を許可します。
ATMOBI PCは`192.168.0.100/24`、Jetsonは`192.168.0.200/24`です。
USB側はそれぞれ`192.168.1.1/24`、`192.168.1.100/24`です。
Fast DDSは上記4アドレスとloopbackのインターフェースだけを使います。
ドメイン番号だけでは社内LANへの通信を防げません。
Fast DDSのインターフェース制限とunicast探索を併用します。
接続元IPを制限するファイアウォールとは異なります。
2台専用LANを社内LANへブリッジしないでください。
ATMOBI PCのGW値は`192.168.0.1`です。
ただし、2台だけの直接接続ではGWは存在しません。
2台の通信にGWは不要なため、JetsonのLAN側にはGWを追加しません。
NICの設定と切り替え手順は、隣接リポジトリの
[`nakanoshima_2026/README.md`](../nakanoshima_2026/README.md)に記載しています。

通常は接続を切り替えても、DDSの設定ファイルを変更しません。
カメラと本PCのブリッジも、同じ両経路対応プロファイルを使います。
ノード起動後に追加したNICで通信できない場合は、
アドレスを有効化してからカメラ・nav・ブリッジを再起動します。
保存設定を変更しただけでは、稼働中のnavコンテナは切り替わりません。
使用するLANまたはUSBを接続し、停止状態で次のコマンドを実行してください。
電源を切って移設した場合に備え、policyも起動対象に含めています。

```bash
docker compose -f docker-compose.jetson.yml -f docker-compose.jetson-humble.yml \
  up -d --force-recreate policy nav
```

2026年10月22日は、LAN・USB両経路対応を稼働中の環境にも反映しました。
policy・navの未起動状態から、ブラウザーと同じ開始処理で起動を確認しました。
USB経由でenableの開始・停止とstatusによる状態確認を実施しました。
標準設定への集約後も、本PCのROS 2側とROS 1側で非ゼロ指令を121件ずつ受信しました。
両側で速度値と受信順序が一致しました。
最大値は並進約0.05 m/s、角速度約0.057 rad/sで、設定した上限内でした。
実機LANでの速度伝達と実走行は未確認です。

ATMOBI PCのブラウザーに、SSH経由の「OmniVLA開始」「OmniVLA終了」を追加しています。
開始操作は必要なpolicy・navを起動し、enableを送ってstatusを確認します。
終了操作はenableの`false`を送って停止状態を確認します。
`NAV_AUTOSTART=false`を維持してください。
パスワードはブラウザーで操作ごとに入力し、アプリには保存しません。
操作手順とホスト鍵の登録は、上記の隣接リポジトリのREADMEを参照してください。

今回のJetson構成は、元からある設定ファイルを編集して使用します。
トピックは`configs/robot.yaml`で設定します。
速度上限と到達判定は`configs/navigator.yaml`で設定します。
旧試験用の`*_usb_test.yaml`は削除せず、リポジトリ外へ退避しました。
退避先は`~/Documents/nakanoshima/.retired-usb-config-20261022-1137/`です。
現構成では旧試験用ファイルを参照しません。

2026年10月22日、`jetson`ブランチで追加した接続試験用のゴール・経路設定を
`master`の既定値へ戻しました。`master`には実コースのゴール画像や経路は登録されておらず、
`.env.example`の`TOPOMAP`は空です。実機の`.env`も空に戻しています。
試験画像`data/topomaps/usb_camera_test/0.jpg`は削除せず残しますが、既定では選択しません。
以前の`reach_check=none`は廃止し、`engine.tracker.reach_check=auto`へ戻しました。
topomapを指定するまでは走行を開始しません。

```bash
ROBOT_CONFIG=/workspace/configs/robot.yaml
NAV_CONFIG=/workspace/configs/navigator.yaml
TOPOMAP=
NAV_AUTOSTART=false
```

`robot.yaml`の`topics.odom`も`master`の`/odom`へ戻しました。
到達判定`auto`は、受信済みの自己位置・オドメトリとtopomapの位置情報に応じて
`pose`、`image_odom`、`image`を選びます。トピック名を戻しただけで、
ATMOBIからJetsonへオドメトリが新たに配線されるわけではありません。
到達判定・ゴール半径等は`master`と同じです。
速度上限もユーザーの指定に従い、`master`と同じ
`engine.controller.track_max_v=0.6 m/s`、
`engine.controller.track_max_w=1.0 rad/s`へ戻しました。
`NAV_AUTOSTART=false`と専用速度トピックへのremapも維持します。
画像の許容経過時間は、元の`master`と同じ0.5秒です。
推論頻度の設定値は0で、新しい画像の到着に追従します。
推論画像のログ保存も、元の`master`と同じ有効設定です。
`robot.yaml`の`topics.cmd`は元の`/cmd_vel`を維持します。
実際のROS 2出力は、起動時のremapで`/omnivla/cmd_vel/raw`に限定します。
今回のUSBカメラは、640×480のYUYV形式で15 FPSを使用します。
Jetsonの`/etc/default/nakanoshima-humble-camera`に`FRAMERATE=15.0`を設定します。
実機が15 FPSで撮像する状態で10 FPSずつ読み出すと、画像が滞留します。
読み出し周期は、カメラが実際に使用するフレームレートに合わせてください。
設定の変更とカメラの再起動後、画像到着時の遅れは約1秒から約0.22秒へ短縮しました。
Humble版は引き続き、`policy_url`で別コンテナの推論サーバーを使用します。
速度トピックのremapとLAN・USB両経路のDDS設定も維持します。
開始と停止は`/omnivla/enable`の`std_msgs/msg/Bool`で操作します。
推論結果が1秒間更新されない場合は、ゼロ速度を出力します。
この設定は速度伝達の接続確認用で、経路の到達性能を確認するものではありません。
実機では、非常停止できる状態で確認してください。

設定はJetson上で編集してください。
本PC側の同名ファイルだけを編集しても、Jetsonには反映されません。
実機の走行とOmniVLAの推論を停止してから設定を変更します。
`.env`の参照先を変更した場合は、navコンテナを再作成します。

```bash
docker compose -f docker-compose.jetson.yml -f docker-compose.jetson-humble.yml \
  up -d --force-recreate nav
```

以降、YAML内の数値だけを変更した場合は、`restart nav`で反映できます。
再ビルドは不要です。
反映後はブラウザーからOmniVLAを開始します。
「OmniVLA終了」は実機全体の非常停止ではありません。

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
