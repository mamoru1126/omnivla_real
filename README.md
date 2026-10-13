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
- まだのこと: Jetson 上でのイメージのビルドと実行、7B の Jetson での推論時間、実機での走行。

7B の Jetson での推論時間は、Jetson で `bin/desk_eval.sh` を実行すると report.txt の「推論時間」に出ます。遅すぎる場合は edge に切り替えます (学習・走行の手順は同じ)。
