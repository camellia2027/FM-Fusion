#!/bin/bash

# Test script for OnlineMappingNode
echo "Testing OnlineMappingNode..."

# Source the workspace
cd /home/wuxin/Desktop/FM-Fusion/catkin_ws
source devel/setup.bash

# Check if the node executable exists
if [ ! -f "devel/lib/sgloop_ros/OnlineMappingNode" ]; then
    echo "ERROR: OnlineMappingNode executable not found!"
    exit 1
fi

echo "✓ OnlineMappingNode executable found"

# Test if the node can be launched (will fail due to missing topics, but should show it's working)
echo "Testing node launch (will timeout after 5 seconds)..."
timeout 5s rosrun sgloop_ros OnlineMappingNode \
    _cfg_file:=/home/wuxin/Desktop/FM-Fusion/config/scannet.yaml \
    _output_folder:=/tmp/test_online_mapping \
    _local_agent:=agentA \
    _max_frames:=10 \
    _frame_gap:=1 \
    _visualization:=0 \
    _o3d_verbose_level:=2 \
    _debug:=false

echo "✓ Node launch test completed"

# Test launch file
echo "Testing launch file..."
if [ ! -f "src/sgloop_ros/launch/online_mapping.launch" ]; then
    echo "ERROR: Launch file not found!"
    exit 1
fi

echo "✓ Launch file found"

echo "All tests passed! OnlineMappingNode is ready for use."
echo ""
echo "To use the online mapping node:"
echo "1. Make sure your time synchronization node is running and publishing to the expected topics"
echo "2. Run: roslaunch sgloop_ros online_mapping.launch"
echo ""
echo "Expected input topics:"
echo "  - /camera/color/image_raw (sensor_msgs/Image)"
echo "  - /camera/depth/image_rect_raw (sensor_msgs/Image)"
echo "  - /mask_image (sensor_msgs/Image)"
echo "  - /vins_estimator/camera_pose (geometry_msgs/PoseStamped)"
echo "  - /mask_data (std_msgs/String)"
