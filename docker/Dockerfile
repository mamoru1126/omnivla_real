# syntax=docker/dockerfile:1
# =============================================================================
# 学習・変換・机上評価用 (x86_64, NVIDIA GPU: RTX 3090 以上)
#   Ubuntu 22.04 / Python 3.10 / ROS 2 Humble (bag 再生・ノードの動作確認用) / PyTorch 2.2.0 (CUDA 12.1)
#   OmniVLA (コミット固定, 最小限の依存) / OmniVLA-edge / rosbags (ROS1・ROS2 の bag を ROS 無しで読む)
# =============================================================================
ARG BASE_IMAGE=ros:humble-ros-base-jammy
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TOKENIZERS_PARALLELISM=false OMNIVLA_ROOT=/opt/OmniVLA OMNIVLA_REAL_ROOT=/workspace \
    PYTHONPATH=/workspace:/opt/OmniVLA
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
        git git-lfs curl ca-certificates build-essential python3-pip python3-dev \
        python3-colcon-common-extensions libgl1 libglib2.0-0 \
        ros-humble-rqt-image-view ros-humble-rosbag2-storage-mcap \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install -U "pip>=24,<26" "setuptools>=68,<70" wheel \
    && python3 -m pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121

COPY docker/requirements.txt docker/install_omnivla.sh /tmp/
RUN echo -e "torch==2.2.0\ntorchvision==0.17.0" > /tmp/keep.txt \
    && python3 -m pip install -c /tmp/keep.txt -r /tmp/requirements.txt \
    && bash /tmp/install_omnivla.sh

COPY . /workspace
WORKDIR /workspace
RUN source /opt/ros/humble/setup.bash && mkdir -p /opt/omnivla_ws \
    && colcon --log-base /opt/omnivla_ws/log build --symlink-install --base-paths /workspace/ros2 \
        --build-base /opt/omnivla_ws/build --install-base /opt/omnivla_ws/install

# 動作確認: import と単体テスト
RUN python3 -c "import torch, transformers, peft, rosbags, clip, efficientnet_pytorch; \
import omnivla_real.policy, omnivla_real.edge; \
print('torch', torch.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__)" \
    && python3 -m pytest -q tests

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
