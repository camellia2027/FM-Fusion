# 在线地图绘制与可视化指南

本指南讲解如何使用新的在线地图绘制启动文件，实现带可视化的实时语义地图绘制。

export LIBGL_ALWAYS_SOFTWARE=true
source devel/setup.bash

## 可用启动文件

### 1. 基础在线地图绘制
```bash
roslaunch sgloop_ros online_mapping.launch
```
- 仅启动在线地图节点
- 不包含可视化
- 适合无头（headless）运行

### 2. 带可视化的在线地图绘制
```bash
roslaunch sgloop_ros online_mapping_with_viz.launch
```
- 启动在线地图节点 + RViz
- 实时显示地图绘制进度
- 推荐用于开发和调试

### 3. 完整在线系统
```bash
roslaunch sgloop_ros complete_online_system.launch play_bag:=true
```
- 启动在线地图 + RViz + 回放bag文件
- 完整端到端系统
- 演示的理想选择

### 4. 独立组件启动

#### 仅启动可视化：
```bash
roslaunch sgloop_ros online_visualize.launch
```

#### 仅启动bag回放：
```bash
roslaunch sgloop_ros play_bag.launch bag_file:=path/to/your.bag
```

## 使用示例

### 示例 1：基于实时数据的地图绘制（带可视化）
```bash
# 终端1：启动带可视化的在线地图绘制
roslaunch sgloop_ros online_mapping_with_viz.launch

# 终端2：发布传感器数据或播放bag文件
rosbag play your_sensor_data.bag --clock
```

### 示例 2：带自定义参数的完整系统
```bash
roslaunch sgloop_ros complete_online_system.launch \
    play_bag:=true \
    bag_file:=../data/ScanNet/newdata.bag \
    bag_rate:=0.5 \
    max_frames:=50 \
    output_folder:=output/my_experiment
```

### 示例 3：离线分析模式
```bash
# 启动无可视化地图绘制，加快处理速度
roslaunch sgloop_ros online_mapping.launch \
    max_frames:=1000 \
    output_folder:=output/batch_processing

# 以更高速度播放bag文件
roslaunch sgloop_ros play_bag.launch \
    bag_file:=data/ScanNet/newdata.bag \
    rate:=2.0
```

## 启动文件参数

### 通用参数
- `cfg_file`：配置文件路径（默认：online.yaml）
- `output_folder`：结果保存路径
- `local_agent`：代理名称（默认：agent0）
- `max_frames`：最大处理帧数
- `verbose_level`：日志详细等级（0-3）

### 可视化参数
- `enable_rviz`：启用/禁用RViz（true/false）

### Bag回放参数
- `play_bag`：启用bag回放（true/false）
- `bag_file`：bag文件路径
- `bag_rate`：回放速度倍率
- `bag_start`：回放起始时间（秒）
- `loop`：循环回放（true/false）

## 可视化功能特点

在线地图绘制可视化包括：

- **实时实例生成**：检测到新实例时立即显示
- **增量点云**：地图逐帧增长展示
- **语义标签**：实例按语义类别着色
- **代理轨迹**：跟踪相机/机器人运动路径
- **实例中心点**：显示实例质心的可视标记
- **边连接关系**：显示实例间空间关系

## 故障排查

### 无可视化显示
1. 检查RViz是否运行：`rosnode list | grep rviz`
2. 验证话题是否发布：`rostopic list`
3. 确认RViz配置文件存在：`online_mapping.rviz`

### 地图节点无数据接收
1. 检查话题名称是否与bag文件匹配：`rostopic list`
2. 确认时钟信息已发布：`rostopic echo /clock`
3. 检查bag回放速度是否过快

### 性能问题
1. 降低bag回放速度：`bag_rate:=0.2`
2. 禁用可视化：`enable_rviz:=false`
3. 限制最大帧数：`max_frames:=100`

## 最佳使用建议

1. **慢速开始**：初期使用`bag_rate:=0.5`确保处理稳定
2. **监控资源**：观察地图绘制时CPU和内存占用
3. **保存配置**：创建自定义启动文件以便重复实验
4. **配置匹配**：实时用online.yaml，精确性要求用scannet_offline.yaml

## 与现有工作流程的整合

新在线地图绘制启动文件设计用于补充现有的离线地图绘制流程：

```bash
# 步骤1：实时地图绘制与可视化
roslaunch sgloop_ros complete_online_system.launch play_bag:=true

# 步骤2：如有需要，转换为ScanNet格式以供离线分析
python3 scripts/bag_to_scannet_format.py data/ScanNet/newdata.bag data/ScanNet/scans

# 步骤3：离线地图绘制对比
roslaunch sgloop_ros semantic_mapping.launch sequence_name:=scene_from_online_bag_fixed
```

该流程实现了从实时地图绘制到详细离线分析的完整管线。