#!/usr/bin/env bash
# 走行中のナビゲーションノードを操作する (bin/jetson.sh up のあと. Jetson の上で).
# .env の JETSON_ROS_DISTRO (humble / noetic) に合わせて ros2 / rostopic を nav コンテナの中で打つ.
#
#   bin/omni.sh start                   # 開始   (/omnivla/enable true)
#   bin/omni.sh stop                    # 停止   (/omnivla/enable false)
#   bin/omni.sh status                  # 状態   (/omnivla/status を 1 回: running / サブゴール番号 / 推論時間)
#   bin/omni.sh hz /camera/compressed   # トピックの周期 (Ctrl-C で終わり)
#   bin/omni.sh topomap /data/topomaps/course_a   # コースを指定して開始 (/omnivla/topomap)
#   bin/omni.sh ros2 topic list         # その他のコマンド (noetic なら rostopic list など)
#   bin/omni.sh sh                      # nav コンテナに入る
# 短く打つなら: sudo ln -sf "$PWD/bin/omni.sh" /usr/local/bin/omni   → omni start
source "$(dirname "$0")/_lib.sh"

DISTRO=$(env_get JETSON_ROS_DISTRO noetic)
# nav コンテナの中で 1 コマンド (ROS の環境を読み込んでから)
in_nav() {
    local setup
    if [ "$DISTRO" = humble ]; then
        setup="source /opt/omnivla_ws/install/setup.bash"
    else
        setup="source /opt/ros/noetic/setup.bash; [ -f /opt/catkin_ws/devel/setup.bash ] && source /opt/catkin_ws/devel/setup.bash"
    fi
    jetson exec "${TTY[@]}" nav bash -c "$setup; $*"
}
pub_bool() {
    if [ "$DISTRO" = humble ]; then
        in_nav "ros2 topic pub --once /omnivla/enable std_msgs/msg/Bool '{data: $1}'"
    else
        in_nav "rostopic pub -1 /omnivla/enable std_msgs/Bool 'data: $1'"
    fi
}

CMD=${1:-}
shift || true
case $CMD in
    start) pub_bool true ;;
    stop) pub_bool false ;;
    status)
        if [ "$DISTRO" = humble ]; then in_nav "ros2 topic echo --once /omnivla/status"; else in_nav "rostopic echo -n 1 /omnivla/status"; fi
        ;;
    hz)
        [ $# -ge 1 ] || die "トピックを指定してください (例 bin/omni.sh hz /camera/compressed)"
        if [ "$DISTRO" = humble ]; then in_nav "ros2 topic hz $1"; else in_nav "rostopic hz $1"; fi
        ;;
    topomap)
        [ $# -ge 1 ] || die "topomap を指定してください (例 bin/omni.sh topomap /data/topomaps/course_a)"
        if [ "$DISTRO" = humble ]; then
            in_nav "ros2 topic pub --once /omnivla/topomap std_msgs/msg/String '{data: $1}'"
        else
            in_nav "rostopic pub -1 /omnivla/topomap std_msgs/String 'data: $1'"
        fi
        ;;
    ros2 | rostopic | rosnode | ros) in_nav "$CMD $*" ;;
    sh) jetson exec nav bash ;;
    *) usage 1 ;;
esac
