"""実機ロボット用 OmniVLA: rosbag -> 学習データ, サブゴール画像列, 机上評価, 走行 (ROS1/ROS2 ノードの中身).

torch / rosbags / ROS は使うモジュールの中で import する (変換・評価だけなら GPU も ROS も不要)。
"""
