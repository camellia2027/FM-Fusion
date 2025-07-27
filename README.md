source devel/setup.bash && ./devel/lib/sgloop_ros/OnlineMappingNode _cfg_file:=/home/wuxin/Desktop/FM-Fusion/config/scannet.yaml _local_agent:=agent0 _output_folder:=/home/wuxin/Desktop/FM-Fusion/output/online_test_final _max_frames:=45

source devel/setup.bash && ./devel/lib/sgloop_ros/MappingNode _cfg_file:=/home/wuxin/Desktop/FM-Fusion/config/scannet_offline.yaml _max_frames:=4500

source devel/setup.bash && rosbag play ../data/ScanNet/newdata.bag --clock

python3 visualize_results.py --result_dir output/online_test_with_mapping/online_mapping --mode both

python3 visualize_results.py --result_dir /home/wuxin/Desktop/FM-Fusion/output/offline_original_correct_config/scene0025_00 --mode both
## 目录
1. [安装](#1-安装)
2. [数据下载](#2-数据下载)
3. [在线语义建图](#3-在线语义建图)
4. [离线语义建图](#4-离线语义建图)
5. [Python脚本使用](#5-Python脚本使用)
6. [配置文件](#6-配置文件)
7. 与你的RGB-D相机配合使用
8. [问题排查](#8-问题排查)
9. [引用](#9-引用)
10. [致谢](#10-致谢)
11. [许可证](#11-许可证)

## 1. 安装

### 依赖项
从Ubuntu源安装依赖包：
```bash
sudo apt-get install libboost-dev libomp-dev libeigen3-dev
```

### Open3D安装
从源码安装Open3D（[安装教程](https://www.open3d.org/docs/release/compilation.html#compilation)）：
```bash
git clone https://github.com/isl-org/Open3D
cd Open3D
mkdir build && cd build
cmake -DBUILD_SHARED_LIBS=ON ..
make -j12
sudo make install
```

### 其他依赖项
按照官方教程安装：
- [OpenCV](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html)（为兼容ROS，推荐OpenCV 3.4.xx版本）
- [GLOG](https://github.com/google/glog)
- [jsoncpp](https://github.com/open-source-parsers/jsoncpp/blob/master/README.md)

### 构建FM-Fusion
```bash
git clone git@github.com:HKUST-Aerial-Robotics/FM-Fusion.git
cd FM-Fusion
mkdir build && cd build
cmake .. -DINSTALL_FMFUSION=ON
make -j12
make install
```

### ROS安装（用于可视化）
按照[官方指南](http://wiki.ros.org/noetic/Installation/Ubuntu)安装ROS，然后构建ROS工作空间：
```bash
git submodule update --init --recursive
cd catkin_ws && catkin_make
source devel/setup.bash
```

## 2. 数据下载

### SgSlam数据集
* [SgSlam_OneDrive](https://hkustconnect-my.sharepoint.com/:f:/g/personal/cliuci_connect_ust_hk/EnIjnIl7a2NBtioZFfireM4B_PnZcIpwcB-P2-Fnh3FEPg?e=BKsgLQ)
* [SgSlam_坚果云](https://www.jianguoyun.com/p/DSM4iw0Q4cyEDRjUvekFIAA)

### ScanNet数据集
* [ScanNet_OneDrive](https://hkustconnect-my.sharepoint.com/:f:/g/personal/cliuci_connect_ust_hk/EhUu2zzwUGNKq8h2yUGHIawBpC8EI73YB_GTG9BUXXBoVA?e=o4TfsA)
* [ScanNet_坚果云](https://www.jianguoyun.com/p/DVuHdogQ4cyEDRjQvekFIAA)

### 数据准备
下载后，解压数据：
```bash
# 先编辑脚本中的路径
python scripts/uncompress_data.py
```

查看[数据格式文档](doc/DATA.md)了解详细的数据结构。

## 3. 在线语义建图

在线建图处理实时数据流（bag文件或实时相机馈送）。

### 3.1 基本在线建图

#### 带ROS可视化
```bash
# 终端1：启动RViz
roslaunch sgloop_ros visualize.launch

# 终端2：运行在线建图（带bag文件）
roslaunch sgloop_ros online_mapping.launch \
    bag_file:=/path/to/your/data.bag \
    config_file:=/path/to/config/online.yaml \
    output_folder:=/path/to/output \
    max_frames:=1000
```

#### 不带可视化（无头模式）
```bash
# 直接执行
rosrun sgloop_ros OnlineMappingNode \
    _cfg_file:=/path/to/config/online.yaml \
    _output_folder:=/path/to/output \
    _max_frames:=1000
```

### 3.2 高级在线建图选项

#### 慢速回放（用于调试）
```bash
# 使用慢速回放脚本进行逐帧分析
./scripts/test_slow_playback.sh
```

#### 调试模式
```bash
roslaunch sgloop_ros debug_online_mapping.launch
```

### 3.3 将Bag文件转换为ScanNet格式
```bash
# 将ROS bag转换为ScanNet格式以便处理
./scripts/convert_bag_to_scannet.sh /path/to/input.bag /path/to/output scene_name
```

## 4. 离线语义建图

离线建图处理带有完整序列的预录制数据集。

### 4.1 基本离线建图

#### 带ROS可视化
```bash
# 终端1：启动RViz
roslaunch sgloop_ros visualize.launch

# 终端2：运行语义建图
roslaunch sgloop_ros semantic_mapping.launch \
    dataroot:=/path/to/dataset \
    sequence_name:=scene0025_00
```

#### 不带可视化
```bash
# 直接执行C++可执行文件
./build/src/IntegrateInstanceMap \
    --config config/scannet.yaml \
    --root /path/to/dataset/scans/scene0025_00 \
    --output /path/to/output
```

### 4.2 批量处理
```bash
# 处理多个序列
python scripts/run_instance_integration.py
```

### 4.3 渲染先前结果
```bash
# 可视化先前计算的结果
roslaunch sgloop_ros render_semantic_map.launch \
    result_folder:=/path/to/results
```

## 5. Python脚本使用

### 5.1 数据处理脚本

#### `scripts/uncompress_data.py`
解压下载的数据集归档文件。
```bash
python scripts/uncompress_data.py
# 先编辑脚本设置你的数据目录
```

#### `scripts/bag_to_scannet_format.py`
将ROS bag文件转换为ScanNet格式。
```bash
python scripts/bag_to_scannet_format.py /path/to/input.bag /path/to/output --scene_name scene_name
```

#### `scripts/read_newdata_bag.py`
分析并从bag文件中提取数据。
```bash
python scripts/read_newdata_bag.py --bag_file /path/to/data.bag --output_dir /path/to/output
```

### 5.2 可视化脚本

#### `scripts/visualize.py`
使用支持WebRTC的Open3D进行3D可视化。
```bash
python scripts/visualize.py --map_dir /path/to/instance_map.ply
```

#### `scripts/rerun_visualize.py`
使用Rerun进行高级可视化。
```bash
# 本地可视化
python scripts/rerun_visualize.py --src_scene_dir /path/to/scene --viz_mode 1

# 远程可视化
python scripts/rerun_visualize.py --src_scene_dir /path/to/scene --viz_mode 2 --remote_rerun_add IP:PORT

# 保存可视化数据
python scripts/rerun_visualize.py --src_scene_dir /path/to/scene --viz_mode 3
```

### 5.3 处理脚本

#### `scripts/run_instance_integration.py`
批量处理多个序列以进行实例整合。
```bash
python scripts/run_instance_integration.py
# 编辑脚本配置数据集路径和参数
```

### 5.4 Shell脚本

#### `scripts/test_slow_playback.sh`
使用慢速bag回放测试在线建图，避免丢帧。
```bash
./scripts/test_slow_playback.sh
```

#### `scripts/convert_bag_to_scannet.sh`
用于bag到ScanNet转换的包装脚本。
```bash
./scripts/convert_bag_to_scannet.sh /path/to/input.bag /path/to/output scene_name
```

## 6. 配置文件

### 6.1 在线建图配置

#### `config/online.yaml`
用于实时处理的基本在线建图配置。
- 针对实时性能进行优化
- 为提高速度采用较低质量设置

#### `config/online_fixed.yaml`
具有更好质量设置的增强型在线建图。
- 更高质量的重建
- 可能需要更多计算资源

### 6.2 离线建图配置

#### `config/scannet.yaml`
用于ScanNet数据集处理的配置。
- 针对ScanNet数据格式进行优化
- 更高分辨率设置

#### `config/realsense.yaml`
用于英特尔实感相机数据的配置。
- 相机特定参数
- 针对实感D515/L515进行优化

### 6.3 关键配置参数

```yaml
# 相机参数
image_width: 640
image_height: 480
camera_fx: 451.08
camera_fy: 600.34

# 建图参数
Mapping:
  voxel_length: 0.02          # 重建体素大小
  sdf_trunc: 0.04             # SDF截断距离
  depth_max: 4.0              # 最大深度范围
  min_det_masks: 60           # 最小检测掩码数
  min_iou: 0.3                # 关联的最小交并比
```

## 7. 与你的RGB-D相机配合使用

### 7.1 硬件设置
我们推荐使用带IMU的英特尔实感D515/L515进行位姿估计。

### 7.2 数据收集
1. 使用[VINS-Mono](https://github.com/HKUST-Aerial-Robotics/VINS-Mono)进行视觉-惯性里程计
2. 将相机位姿保存在`pose`文件夹中
3. 收集标准格式的RGB-D图像

### 7.3 实例分割
运行[Grounded-SAM](https://github.com/glennliu/Grounded-Segment-Anything)生成：
- 检测结果（`frame-XXXXXX_label.json`）
- 实例掩码（`frame-XXXXXX_mask.png`）

### 7.4 处理你的数据
```bash
# 使用你的自定义配置
./build/src/IntegrateInstanceMap \
    --config config/your_camera.yaml \
    --root /path/to/your/data \
    --output /path/to/output
```

## 8. 问题排查

### 8.1 常见问题

#### OnlineMappingNode中的段错误
```bash
# 检查配置文件路径和格式
# 确保所有依赖项都正确安装
# 尝试使用调试模式：
roslaunch sgloop_ros debug_online_mapping.launch
```

#### ROS通信问题
```bash
# 检查ROS主节点
roscore

# 验证话题发布
rostopic list
rostopic echo /your/topic
```

#### 内存问题
```bash
# 减少max_frames或增加系统内存
# 监控内存使用：
htop
```

### 8.2 性能优化

#### 对于实时处理
- 使用`config/online.yaml`，其具有较低质量设置
- 减少`max_frames`参数
- 增加`frame_gap`以跳帧

#### 对于高质量结果
- 使用`config/scannet.yaml`或`config/realsense.yaml`
- 用完整序列进行离线处理
- 确保有足够的计算资源