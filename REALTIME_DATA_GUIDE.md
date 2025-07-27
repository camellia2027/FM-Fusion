# 实时数据集成指南

本指南介绍如何将实时传感器数据与 FM-Fusion 在线地图系统集成。

## 概述

OnlineMappingNode 设计用于通过 ROS 主题接收实时数据。**从袋文件切换到实时数据流无需修改代码**。

## 必要的主题和消息格式

### 主题：`/sync/output`  
**消息类型**：`sgloop_ros/SyncedFrame`

```
Header header                # 时间戳和坐标系ID
sensor_msgs/Image rgb        # RGB图像（bgr8 编码）
sensor_msgs/Image depth      # 深度图像（16UC1 编码）
sensor_msgs/Image mask       # 实例掩码（mono8 编码）
float64[16] transform_matrix # 4x4 位姿矩阵（行优先）
string json                  # 检测结果的 JSON 格式字符串
```

## 数据要求

### 1. RGB 图像
- **编码**：`bgr8`
- **分辨率**：任意（通常为 640x480 或更高）
- **格式**：标准 OpenCV BGR 格式

### 2. 深度图像
- **编码**：`16UC1`
- **单位**：毫米
- **范围**：0-65535（0 表示无效深度）

### 3. 实例掩码
- **编码**：`mono8`
- **数值**：实例 ID（1, 2, 3, ..., 255）
- **背景**：0（无实例）

### 4. 变换矩阵
- **格式**：4x4 齐次变换矩阵
- **顺序**：行优先展平数组（16 个元素）
- **坐标系**：相机在世界坐标系下的位姿

### 5. 检测 JSON
```json
[
    {
        "value": 1,
        "label": "chair",
        "confidence": 0.95,
        "bbox": [x1, y1, x2, y2]
    },
    {
        "value": 2,
        "label": "table",
        "confidence": 0.88,
        "bbox": [x1, y1, x2, y2]
    }
]
```

## 实时数据启动文件

### 1. 基础实时映射
```bash
roslaunch sgloop_ros realtime_mapping.launch
```

### 2. 完整实时系统
```bash
roslaunch sgloop_ros realtime_system.launch
```

### 3. 自定义参数启动
```bash
roslaunch sgloop_ros realtime_system.launch \
    queue_size:=1 \
    max_frames:=0 \
    output_folder:=output/my_realtime_test
```

## 性能优化

### 1. 队列大小
实时处理推荐使用较小的队列大小：
```xml
<param name="subscriber_queue_size" value="1"/>
```

### 2. 处理频率
系统可处理约 ~10 Hz，具体取决于：
- 图像分辨率
- 每帧检测数
- 硬件性能

### 3. 内存管理
- 使用 `max_frames:=0` 实现无限制处理
- 长时间运行时监控内存使用
- 考虑定期重启以保证 24/7 运行稳定

## 集成示例

### 示例 1：相机 + SAM 集成
```python
#!/usr/bin/env python3
import rospy
from sgloop_ros.msg import SyncedFrame
from your_camera_driver import CameraDriver
from your_sam_detector import SAMDetector

class RealtimeMappingBridge:
    def __init__(self):
        self.camera = CameraDriver()
        self.detector = SAMDetector()
        self.pub = rospy.Publisher('/sync/output', SyncedFrame, queue_size=1)
    
    def process_frame(self):
        # 获取相机数据
        rgb, depth, pose = self.camera.get_frame()
        
        # 运行检测
        mask, detections = self.detector.detect(rgb)
        
        # 创建并发布 SyncedFrame
        frame = self.create_synced_frame(rgb, depth, mask, pose, detections)
        self.pub.publish(frame)
```

### 示例 2：RealSense + 自定义检测器
```python
import pyrealsense2 as rs
import numpy as np

class RealSensePublisher:
    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.pipeline.start(config)
        
    def get_frames(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        
        rgb_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        
        return rgb_image, depth_image
```

## 使用示例数据进行测试

### 1. 启动映射系统：
```bash
roslaunch sgloop_ros realtime_system.launch
```

### 2. 运行示例发布者：
```bash
roslaunch sgloop_ros realtime_system.launch enable_sample_publisher:=true
```

### 3. 监控主题：
```bash
rostopic hz /sync/output
rostopic echo /sync/output --noarr
```

## 故障排查

### 无数据接收
1. 检查主题名称：`rostopic list | grep sync`
2. 验证消息类型：`rostopic info /sync/output`
3. 检查发布频率：`rostopic hz /sync/output`

### 处理过慢
1. 减小队列大小：`queue_size:=1`
2. 降低图像分辨率
3. 减少检测频率
4. 禁用可视化：`enable_rviz:=false`

### 内存问题
1. 设置帧数限制：`max_frames:=1000`
2. 使用 `htop` 或 `free -h` 监控
3. 长期运行建议定期重启

## 从袋文件迁移

### 迁移前（袋文件）：
```bash
# 终端1
roslaunch sgloop_ros online_mapping_with_viz.launch

# 终端2  
rosbag play data.bag --clock
```

### 迁移后（实时）：
```bash
# 终端1
roslaunch sgloop_ros realtime_system.launch

# 终端2
python3 your_realtime_publisher.py
```

## 关键区别

| 方面       | 袋文件           | 实时数据         |
|------------|------------------|------------------|
| **时间同步** | 受控回放         | 实时时间戳       |
| **队列大小** | 默认 10          | 推荐 1           |
| **时钟类型** | 仿真时间         | 实时时间（墙钟） |
| **延迟**     | 无               | 尽量减少排队     |
| **可靠性**   | 可重复           | 需处理丢包       |

## 最佳实践

1. **同步**：确保所有数据（RGB、深度、掩码、位姿）时间对齐  
2. **错误处理**：优雅处理丢失或损坏的帧  
3. **性能监控**：监测处理延迟，适当调整速率  
4. **验证**：上线前用已知数据集测试  
5. **日志**：开启适当的日志记录以便调试

系统已准备好接收实时数据 —— 只需发布到 `/sync/output` 即可开始建图！