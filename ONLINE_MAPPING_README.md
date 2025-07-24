# FM-Fusion 在线建图

本文档介绍了 FM-Fusion 的在线建图功能。

## 🎯 概述

在线建图系统允许 FM-Fusion 处理实时同步的数据流，而非离线文件。它订阅一个包含同步的 RGB、深度、掩码、位姿和 JSON 数据的单一 ROS 话题。

## 📋 组件

### 核心文件
- `catkin_ws/src/sgloop_ros/msg/SyncedFrame.msg` - 自定义同步数据消息类型
- `catkin_ws/src/sgloop_ros/src/SimpleOnlineMappingNode.cpp` - 简化版在线建图节点
- `catkin_ws/src/sgloop_ros/src/OnlineMappingNode.cpp` - 完整在线建图节点（含 FM-Fusion 集成）
- `catkin_ws/src/sgloop_ros/launch/online_mapping.launch` - 在线建图启动文件

### 测试与工具
- `bag_to_synced_frame.py` - 将 bag 文件转换为 SyncedFrame 消息用于测试
- `test_simple_online_mapping.sh` - 简易测试脚本
- `test_online_mapping.sh` - 通用测试脚本

## 🚀 使用方法

### 1. 使用你的时间同步节点

你的时间同步节点应发布至 `/synced_frame` 话题，消息格式为 `SyncedFrame`：

```cpp
#include "sgloop_ros/SyncedFrame.h"

sgloop_ros::SyncedFrame msg;
msg.header = your_header;
msg.rgb = your_rgb_image;      // sensor_msgs/Image
msg.depth = your_depth_image;  // sensor_msgs/Image  
msg.mask = your_mask_image;    // sensor_msgs/Image
msg.pose = your_pose;          // geometry_msgs/Pose
msg.json = your_json_string;   // string

publisher.publish(msg);
```

然后启动在线建图节点：
```bash
cd /home/wuxin/Desktop/FM-Fusion/catkin_ws
source devel/setup.bash
roslaunch sgloop_ros online_mapping.launch
```

### 2. 使用 Bag 文件测试

```bash
# 终端 1：启动 roscore
roscore

# 终端 2：启动在线建图节点
rosrun sgloop_ros SimpleOnlineMappingNode \
    _output_folder:/tmp/online_mapping_output \
    _max_frames:50

# 终端 3：转换并发布 bag 数据
python3 bag_to_synced_frame.py data/to_fm.bag 1.0
```

### 3. 自动化测试

```bash
./test_simple_online_mapping.sh
```

## 📊 SyncedFrame 消息格式

```
Header header                # 原始图像时间戳
sensor_msgs/Image rgb        # RGB 图像 (/camera/color/image_raw)
sensor_msgs/Image depth      # 深度图像 (/camera/depth/image_rect_raw)
sensor_msgs/Image mask       # SAM 掩码 (/mask_image)
geometry_msgs/Pose pose      # VINS 位姿
string json                  # SAM JSON 数据 (/mask_data)
```

## ⚙️ 参数说明

### SimpleOnlineMappingNode 参数
- `output_folder`：结果保存目录
- `local_agent`：Agent 名称（默认："agentA"）
- `max_frames`：最大处理帧数（默认：5000）
- `frame_gap`：每隔 N 帧处理一次（默认：1）
- `debug`：开启调试模式（默认：false）

### OnlineMappingNode 参数（完整版）
- 包含所有 SimpleOnlineMappingNode 参数，另外还包括：
- `cfg_file`：FM-Fusion 配置文件路径
- `visualization`：是否开启可视化（默认：0）
- `o3d_verbose_level`：Open3D 日志详细级别（默认：2）

## 🔄 版本控制

- **当前分支**：`online-input-feature`
- **备份标签**：`v1.0-offline`（原离线版本）
- **远程仓库**：[https://github.com/camellia2027/FM-Fusion.git](https://github.com/camellia2027/FM-Fusion.git)

切换回离线版本：
```bash
git checkout v1.0-offline
```

## 🎯 集成说明

1. **无需时间同步模块**：系统假定你的时间同步节点已完成时间同步  
2. **消息格式兼容性**：SyncedFrame 设计与 vins_estimator/SyncData 格式匹配  
3. **依赖简洁**：只需标准 ROS 包，无需额外 JSON 库  
4. **易于扩展**：可轻松切换 SimpleOnlineMappingNode（测试）与 OnlineMappingNode（完整版）

## 🐛 故障排查

1. **节点崩溃**：先用 SimpleOnlineMappingNode 验证数据流是否正常  
2. **消息类型错误**：确保发布者严格使用 SyncedFrame 格式  
3. **无数据接收**：检查话题名及消息发布频率  
4. **性能问题**：调整 `frame_gap` 和 `max_frames` 参数