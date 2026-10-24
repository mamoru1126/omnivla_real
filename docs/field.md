# 実機の日の手順 (Orin: ROS 2 Humble, ネットなし)

全部 bash スクリプトを打つだけです。Orin では `bin/field.sh`、作業 PC では `bin/orin.sh` (どちらも `-h` で使い方)。
イメージの作り直しは要りません (変わるのは Python のコードとスクリプトだけで、コンテナにはリポジトリをそのまま見せています)。

## 前の晩 (作業 PC, ネットがあるうちに)

```bash
cd omnivla_real && git checkout jetson && git pull
```

`~/.ssh/config` に Orin を書いておくと、以降は `orin` だけで届きます (ポートが 22 以外でも)。

```
Host orin
    HostName 192.168.0.200
    User <Orin のユーザー名>
    Port 4003
```

## 当日

### 0. Orin を更新する (作業 PC で. LAN でつながっていれば)

```bash
bin/orin.sh sync                 # 作業 PC のリポジトリ (jetson) を Orin に送って取り込む
```

LAN で届かない場合は USB で: 作業 PC で `bin/orin.sh bundle` → できた `omnivla_jetson.bundle` を Orin へ →
Orin のリポジトリで `git fetch <ファイル> jetson && git merge --ff-only FETCH_HEAD`。

### 1. 起動と確認 (Orin で. 作業 PC からなら `bin/orin.sh ssh` で入る)

```bash
bin/field.sh setup --topomap course_a_144933   # 最初の 1 回: .env に走行用の値を書く
bin/field.sh time                              # ATMOBI PC との時計のずれ → ずれていれば合わせる
bin/field.sh up --max-v 0.3                    # 推論サーバ + ナビゲーション + デバッグ画面. 準備できるまで待つ (走り出さない)
bin/field.sh check --where                     # 全部の確認. NG が無ければ次へ
```

- デバッグ画面: http://192.168.0.200:8090 (ATMOBI PC のブラウザから)。カメラ画像と予測軌跡、サブゴール画像、状態。
- `check --where` の最後の行で、いまの画像が topomap の何番に似ているかが分かります。スタート位置に置いたら 0 番の近く、類似度 0.9 以上が目安。

### 2. 手で走らせて、モデルが付いてくるか見る (指示値はロボットにつながない)

```bash
bin/field.sh record manual       # 記録を始める (裏で)
bin/field.sh start               # 推論を開始 (数秒後の状態も表示)
#   … ATMOBI PC のジョイスティックでコースを 1 周. デバッグ画面でサブゴールが進むか、予測軌跡がコースに沿うか見る
bin/field.sh stop
bin/field.sh record stop
bin/field.sh summary             # まとめ (状態, サブゴール, 類似度, 推論時間, v/w, 画像の遅れ)
```

### 3. 低速で自律走行

ATMOBI PC 側で `/omnivla/cmd_vel/raw` を `/cmd_vel` に切り替えてから。非常停止をすぐ押せるように。

```bash
bin/field.sh record auto1
bin/field.sh start
bin/field.sh stop                # 止める (デバッグ画面の停止ボタン / Esc でも)
bin/field.sh record stop
```

### 4. 片付けと持ち帰り

```bash
bin/field.sh down                # ナビゲーション・デバッグ画面・記録を止める (推論サーバはそのまま. 全部: down --all)
# 作業 PC で
bin/orin.sh fetch                # Orin の記録を debug_log/orin_<日時>/ へ (bag 本体と画像を除く. 要るなら --bags --images)
```

LAN が無い場合は Orin で `bin/field.sh collect /media/<USB>` (bag と画像を除いた tar.gz)。

## 実走を自己位置で評価する (自己位置は評価だけ. 走行には使わない)

最初に 1 回、自己位置のトピック (PoseStamped) を robot.yaml に書いておくと、`record` で一緒に録ります。

```bash
bin/field.sh topic eval_localization /<自己位置のトピック>
```

自律走行を録ったあと (コースの topomap は自己位置の入った bag から作ったもの = 地図の座標):

```bash
bin/run_eval.sh log/bags/auto3_<時刻> course_b
```

- `runs/run_eval/<bag>_<時刻>/report.txt`: どこまで進んだか、コースからの横ずれ、カーブごとに外へ膨らんだ量。`/omnivla/status` で自律で走っていた間だけの集計も出ます。
- `overview.png` (走った所を横ずれの大きさで色分け)、`xtrack.png`。作業 PC に持ってくるなら `bin/orin.sh logs` ではなく `scp` で (画像なので)。

## 学習し直したモデルと新しい topomap を入れる

学習 PC の `runs/<run>/checkpoints/step_XXXXXX/` と `data/topomaps/<名前>/` を、作業 PC のリポジトリの同じ場所にコピーしてから (USB か scp)。

```bash
# 作業 PC で
bin/orin.sh send runs/<run>/checkpoints/step_005000 data/topomaps/course_b
# Orin で
bin/field.sh setup --model runs/<run>/checkpoints/step_005000 --topomap course_b
bin/field.sh down --all
bin/field.sh up --max-v 0.3
```

- `down --all` で推論サーバも止めて、`up` で新しいモデルを読み込ませる (推論サーバは起動したときのモデルのまま)。
- 曲がりが足りなければ `--w-gain`。目安は 1 / (机上評価の「曲がる量の比」)。0.8 なら `--w-gain 1.2`。

## 困ったとき

| 症状 | 確認 | 対処 |
|---|---|---|
| 指示値が 0 で類似度が null | `bin/field.sh status` が `step_state=waiting_image` | `bin/field.sh check` で画像の遅れを見る。`up` は既定で受け取ってからの時間で判定するので、時計のずれでは止まらない |
| カメラ画像が届かない (check で NG) | 似た名前のトピックが表示される | `bin/field.sh topic image <名前>` → `bin/field.sh up` |
| `start` で `reach_check=image` と出る / サブゴールが走っていないのに進む | `check` で「オドメトリが届いていない」 | /odom が届いてから `start` する (遅れて届けば自動で image_odom に切り替わる) |
| 時計がずれている | `bin/field.sh time` | NTP で届かなければ `ATMOBI_SSH=<user>@192.168.0.100 bin/timesync.sh step` |
| 類似度が低い / サブゴールが進まない (`where` で 0.9 未満) | topomap を撮った時とカメラ・景色が違う | 2 で録った bag から topomap を作り直す: `bin/field.sh topomap log/bags/manual_<時刻> course_today` → `bin/field.sh up --topomap course_today` |
| 曲がりが弱い (外に膨らむ) | モデルの予測軌跡そのものが曲がり不足 (机上評価で実際の 7 割ほど) | `bin/field.sh up --max-v 0.3 --w-gain 1.3` (曲がる量を 1.3 倍. 1.5 までで様子を見る). `--max-v` で速度を落としても曲がり方 (曲率) は変わらない |
| 推論が遅い | `status` の latency (7B は 600 ms 前後) | 速度を下げる (`up --max-v 0.2`) |
| ノードの様子 | `bin/jetson.sh logs nav` / `bin/jetson.sh logs policy` | |

- 録った bag で机上評価もできます (`bin/field.sh eval log/bags/manual_<時刻> course_a_144933 --end_sec 120`)。
  Orin では 1 フレーム 1 秒近くかかるので、短く切るか、持ち帰って学習 PC で。
- ATMOBI PC のブラウザの「OmniVLA 開始」から起動しても、`setup` で書いた `.env` の値 (画像の古さの測り方など) が使われます。
