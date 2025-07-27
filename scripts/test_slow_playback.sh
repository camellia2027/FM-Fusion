#!/bin/bash

# 测试慢速播放bag文件以避免丢帧

echo "Starting slow playback test..."

# 启动roscore（如果没有运行）
if ! pgrep -x "roscore" > /dev/null; then
    echo "Starting roscore..."
    roscore &
    sleep 3
fi

# 启动在线建图节点
echo "Starting OnlineMappingNode..."
# 获取脚本目录并构建相对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT/catkin_ws"
source devel/setup.bash

# 使用较大的max_frames来测试所有帧
./devel/lib/sgloop_ros/OnlineMappingNode \
    _cfg_file:="$PROJECT_ROOT/config/scannet.yaml" \
    _local_agent:=agent0 \
    _output_folder:="$PROJECT_ROOT/output/slow_test" \
    _max_frames:=50 &

MAPPING_PID=$!
sleep 5

echo "Starting slow bag playback (rate 0.5x)..."
# 以0.5倍速度播放，给处理更多时间
rosbag play ../data/ScanNet/newdata.bag --clock --rate=0.5

echo "Waiting for mapping node to finish..."
wait $MAPPING_PID

echo "Test completed!"
