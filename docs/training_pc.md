# 学習 PC での手順 (x86_64 + NVIDIA GPU)

[README](../README.md) の 1〜7 です。Jetson での走行 (8) は [jetson.md](jetson.md) にあります。

## 準備

Docker と NVIDIA Container Toolkit が必要です。

```bash
git clone https://github.com/mamoru1126/omnivla_real.git && cd omnivla_real
bin/setup.sh          # .env を作る + イメージのビルド + 公式の重み (7B と edge). edge だけなら bin/setup.sh edge
```

フォルダとコンテナ内のパスの対応 (`.env` で変更可):

| ホスト | コンテナ |
|---|---|
| `./bags` | `/bags` |
| `./data` | `/data` (学習データ・topomap) |
| `./runs` | `/runs` (学習・評価の結果) |
| `./checkpoints` | `/checkpoints` (公式の重み) |

### コマンド (`bin/`)

各段階は `bin/` のスクリプトで実行します。中で `docker compose` を呼ぶので、どこから実行しても構いません。

| 段階 | コマンド | 中でしていること |
|---|---|---|
| 準備 | `bin/setup.sh [all\|edge\|7b\|none]` | `.env` を作る、`docker compose build`、`scripts/download_checkpoints.sh` |
| 1 | `bin/bag_info.sh BAG` | `tools/bag_info.py` |
| 2 | `bin/check_odometry.sh BAG` | `tools/check_odometry.py` |
| 3 | `bin/make_dataset.sh BAG...` | `tools/bag_to_dataset.py` + `training/inspect_dataset.py` |
| 4 | `bin/train.sh 7b\|edge [-d]` | `training/finetune_omnivla.py` / `finetune_edge.py` |
| 5 | `bin/make_topomap.sh 名前 BAG` | `tools/make_topomap.py` |
| 6 | `bin/desk_eval.sh BAG [topomap]` | `tools/desk_eval.py` |
| 7 | `bin/replay.sh BAG [topomap]` / `again` / `down` | `docker-compose.replay.yml` |
| 7 | `bin/replay_video.sh BAG [topomap]` / `--log ログ` | 同じ構成で bag を流し、デバッグ画面と同じ内容を動画に (`tools/nav_video.py`) |
| | `bin/plot_nav_log.sh [ログ]` | `tools/plot_nav_log.py` |
| 8 | `bin/to_jetson.sh user@jetson [--bag BAG]` | 重みと topomap (と評価用の bag) を Jetson に rsync で送る ([jetson.md](jetson.md)) |
| | `bin/shell.sh [コマンド]` | 学習用コンテナに入る / 1 コマンド実行 |

- bag や topomap は、名前 (`run1.bag`, `course_a`)、ホストのパス (`./bags/run1.bag`)、コンテナのパス (`/bags/run1.bag`) のどれで指定しても構いません。分割された bag は `a_0.bag,a_1.bag` です。
- 後ろに付けたオプション (`--start_sec 10` など) は、そのまま中のツールに渡します。
- モデルと topomap は `.env` の `NAV_MODEL` / `FINETUNED_DIR` / `NAV_WEIGHTS` / `TOPOMAP` を使います (机上評価・bag での確認・Jetson で同じ)。
- 使い方は `-h` で表示します。`OMNIVLA_DRY_RUN=1` を付けると、実行せずに中のコマンドを表示します。

## 1. bag の中身を見る

```bash
bin/bag_info.sh run1.bag      # ROS 1 (./bags/run1.bag)
bin/bag_info.sh run1          # ROS 2: bag のディレクトリ (.db3 / .mcap)
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
bin/check_odometry.sh run1.bag     # 結果は runs/check/run1/
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
bin/make_dataset.sh run1.bag run2.bag      # data/dataset に追加. runs/inspect に正解軌跡を画像に重ねた確認用の図
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
bin/train.sh 7b                    # OmniVLA 7B, LoRA (24GB 以上). configs/finetune_7b.yaml
bin/train.sh edge                  # OmniVLA-edge (数 GB の GPU で可). configs/finetune_edge.yaml
bin/train.sh 7b -d                 # 裏で動かす (端末を閉じても続く). ログは docker logs -f omnivla_train_7b
bin/train.sh 7b --max_steps 3000   # 設定ファイルの値をオプションで上書き
```

- 結果は `runs/<run>/checkpoints/step_XXXXXX/` に出ます。走行に使うものを `.env` の `NAV_MODEL` と `FINETUNED_DIR` (例 `/runs/<run>/checkpoints/step_005000`) に書きます。
- 検証に使う走行は `val_bags: [run2]` のように bag 名で指定できます。
- 7B をマージ済みモデルにしたい場合は `training/merge_lora.py` を使います。

### 曲がりが弱い (直進しがち) とき

机上評価の「曲がる量の比」(下の表の `turn_gain`) が 1 より小さいときに見るところです。`configs/finetune_7b.yaml` で設定します。

| 設定 | 中身 |
|---|---|
| `label_smooth_sec` | 正解 (この先の経路) を時間方向にならす [秒]。手動操作でスティックを短く倒して曲がっていると、いつ曲がるかが画像では決まらず、モデルは遅く弱く曲がる。`python3 tools/turn_stats.py /data/dataset` の「倒し切り %」が高ければ 0.5〜1.0 |
| `turn_sample_ratio` | 曲がる場面を引く割合 (0.5 = 半分) |
| `image_goal_dist_m` | ゴール画像を道のりで何 m 先から選ぶか。走行のサブゴールは topomap の 1 m 間隔なので `[0.5, 2.5]` |
| `yaw_aug_deg` | 現在画像を左右にずらし「向きがずれた」状態を作って、元の経路へ戻る正解で学習する (外れたら戻る)。`yaw_aug_hfov_deg` はカメラの実際の水平画角 |

- 学習のログの `data:` の行に、これらの設定と、60 秒ごとの区切りで外したサンプルの数が出ます (区切っただけの所の末尾は「その先で止まる」というウソの正解になるので使いません)。
- 一番効くのはお手本の取り方です。走らせる速さで、なめらかに曲がり、何周か (コースの左寄り・右寄りも) 走ります。自律走行させた bag (モデル自身の動き) は学習に入れません。

## 5. サブゴール画像列 (topomap) を作る

コースを 1 回走った bag から、1 m ごとに画像を切り出します。

```bash
bin/make_topomap.sh course_a run1.bag --spacing 1.0     # data/topomaps/course_a/. .env の TOPOMAP=/data/topomaps/course_a
```

- 出力は `0.jpg, 1.jpg, ...`、`poses.yaml` (各画像の位置)、`overview.png` です。
- 使う範囲は `--start_sec` / `--end_sec` で切り出せます。

## 6. 机上評価

topomap を作った走行とは別の走行の bag で評価します。録画した画像を順にモデルへ入れ、走行時と同じ処理をして、記録と比べます。

- サブゴールの切り替え
- 推論
- 予測軌跡 → 指示値

```bash
bin/desk_eval.sh run2.bag course_a     # モデルは .env の NAV_MODEL / FINETUNED_DIR. 結果は runs/desk_eval/run2_<時刻>/
# 推論サーバ経由 (bin/replay.sh で推論サーバが動いているとき): bin/desk_eval.sh run2.bag course_a --model remote --url http://127.0.0.1:8765
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
| `waypoints.turn_gain` | 曲がる量の比: 予測した 8 点目の向きが実際の何倍か (10 度以上曲がる所)。1 に近いほどよい。0.7 なら 3 割曲がり不足。`turn_lag_frames` が負なら予測が遅れている |
| `integrated_windows` | 実際の位置から N 秒間、モデルの指示値だけで進めたときの終点のずれ。記録の指示値で同じことをした値 (`recorded_cmd`) と同程度なら十分 |
| `subgoals.reached_goal` | 最後のサブゴールまで切り替わったか。`reach_check` は判定の方法 |

- 入力は録画なので開ループです (モデルが曲がり損ねても、画像は記録どおりに進みます)。スタートから積算した軌跡 (`integrated_full`) は誤差が溜まるので、主に N 秒窓で見ます。
- `--policy oracle` を付けると、モデルの代わりに記録の正解を返します。評価やサブゴール切り替えの仕組みの確認に使います。
- `--goal_mode hindsight` にすると、topomap を使わず「記録の 1.5 m 先の画像」をゴールにします。

サブゴールの切り替え方 (`configs/navigator.yaml` の `engine.tracker.reach_check: auto`):

| 状況 | 判定 |
|---|---|
| 自己位置がある | 位置で判定 (`pose`) |
| 自己位置がない | 画像の類似度 + オドメトリの道のり (走った距離) で判定 (`image_odom`)。道のりで先すぎるサブゴールは候補にしない。画像で判定できないまま道のりで通り過ぎたら進める。オドメトリの位置 (x, y) は長いコースだと向きのドリフトで大きくずれるので使わない |

画像の類似度のしきい値 (`image_threshold`) は、机上評価の `report.json` の `similarity_threshold` を見て決めます。

## 7. bag をロボットの代わりにして、実機と同じ構成で動かす

机上評価 (6) とは別に、走行時の構成そのものを学習 PC で動かして確かめます (Jetson の上でも同じコマンドで動きます: [jetson.md](jetson.md#jetson-で机上評価bag-での確認))。Jetson で使う `nav` コンテナ (`docker-compose.jetson.yml` の定義そのまま) と推論サーバを立て、別の走行の bag をロボットの代わりに流します。様子はブラウザのデバッグ画面で見ます。

| コンテナ | 実機では | 中身 |
|---|---|---|
| `roscore` | ロボットのメイン PC など | ROS master |
| `policy` | Jetson の `policy` | 推論サーバ (同じ `tools/policy_server.py`。ここでは x86 の GPU) |
| `nav` | Jetson の `nav` | ROS 1 ナビゲーションノード (C++) とデバッグ画面。**定義もイメージも Jetson と同じ** |
| `robot` | ロボット (カメラ・オドメトリ) | bag の画像・オドメトリ (・自己位置) をシミュレーション時刻で流す (`scripts/bag_robot.sh`) |
| `record` | | ノードが出した `/cmd_vel`, `/omnivla/path`, `/omnivla/status` を記録 |

```bash
# .env に NAV_MODEL / FINETUNED_DIR (Jetson と同じ値) を書いておく
bin/replay.sh run2.bag course_a          # 起動して robot と nav のログを表示 (Ctrl-C でログの表示だけ止まる)
# → ブラウザで http://localhost:8080 (別の PC からは http://<学習 PC の IP>:8080)
bin/replay.sh again                      # もう一度最初から流す (推論サーバはそのまま)
bin/replay.sh down                       # 片付け
```

- `robot` は、`nav` が準備できる (推論サーバのモデルを読み終わる) のを待ってから bag を流し始めます。7B は数分かかります。
- bag からは `robot.yaml` の画像・オドメトリ (・自己位置) だけを流します。記録された `/cmd_vel` は流しません。
- ROS master はこの PC の中だけです (`.env` の `ROS_MASTER_URI` がロボットを指していても使いません)。

| オプション | `.env` | 意味 |
|---|---|---|
| (bag) | `REPLAY_BAG` | bag。カンマ区切りで複数、ディレクトリなら中の `*.bag` 全部 |
| `--rate R` | `REPLAY_RATE` | 再生速度 (既定 1.0) |
| `--start S` | `REPLAY_START` | 先頭から何秒飛ばすか |
| `--args "-u 60"` | `REPLAY_ARGS` | `rosbag play` にそのまま渡す (この例は 60 秒だけ) |
| `--manual` | `NAV_AUTOSTART=false` | ブラウザの「開始」を押してから走り出す (実機と同じ確認ができる) |

- 出力は `RUNS_DIR/replay/` に出ます。
  - `out_<時刻>.bag`: ノードが出した指示値・予測軌跡・状態
  - `nav/<時刻>/`: 走行ログ。`bin/plot_nav_log.sh` で図とレポート
- bag が終わっても画面はそのまま見られます。

**動画に残す** (ブラウザ不要): 同じ構成で bag を最後まで流し、デバッグ画面と同じ内容 (カメラ画像 + 予測軌跡、サブゴール、v / ω、俯瞰図、60 秒の履歴、出来事) を mp4 にします。

```bash
bin/replay_video.sh run2.bag course_a                 # -> runs/replay/<時刻>.mp4
bin/replay_video.sh run2.bag course_a --speed 4       # 動画を 4 倍速で (--rate は bag の再生速度)
bin/replay_video.sh --log runs/replay/nav/latest      # 既にある走行ログから動画だけ作る
bin/replay_video.sh --log log/nav/20261013_120000     # Jetson から持ってきた実機の走行ログも同じ
```

- 動画は走行ログ (`steps.csv`、カメラ画像 `raw/`、サブゴール画像 `goals/`、`events.log`) から作ります。実機のログも `io.log_images: true` (既定) なら動画にできます。
- 初回は ROS 1 のイメージに ffmpeg を入れるため、作り直しが入ります。
- `bin/replay.sh` を使わずに `docker compose -f docker-compose.replay.yml up` (`REPLAY_BAG` と `TOPOMAP` は `.env` か環境変数) でも同じです。
- bag の画像は指示値に反応しません (開ループ)。モデルが曲がり損ねても、画像は記録どおりに進みます。ロボットが指示どおりに動いたらどう見えるか、までは分かりません。

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
bash tests/bin_check.sh                    # bin/ のスクリプトが正しいコマンドを組み立てるか (docker は実行しない)
```

CI (GitHub Actions) では合成したコース走行の bag を使い、次を確認しています。

- ROS 1 / ROS 2 sqlite3 / ROS 2 mcap の bag を読む
- 変換 → edge の学習 (CPU で数 step) → topomap → 机上評価
- 推論サーバ経由の推論が、プロセス内の推論と同じ結果になる
- C++ のノードの中身 (制御・サブゴールの切り替え・走行全体) が Python の実装と同じ結果になる (`tests/test_cpp.py`)
- ROS 1: 推論サーバ + ROS 1 コンテナ (`docker/Dockerfile.ros1`) で bag を再生し、`/cmd_vel` と `/omnivla/path` が出る。デバッグ画面に結果と画像が届く (C++ / Python 両方のノード)
- `docker-compose.replay.yml`: Jetson と同じ `nav` コンテナ + bag をロボット代わりにして、デバッグ画面と記録した出力を確認 (推論サーバは CPU)
- ROS 2 ノードも同様
- 学習用イメージのビルド、Jetson 用推論サーバの Dockerfile (x86 の PyTorch イメージを代わりのベースにして依存のインストールまで)
