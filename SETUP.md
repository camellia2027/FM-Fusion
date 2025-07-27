# FM-Fusion 安装指南

本文档提供了在克隆仓库后进行 FM-Fusion 设置的说明。

## 环境要求

1. **ROS Noetic** - 确保已安装 ROS Noetic  
2. **LibTorch** - 用于深度学习推理的 PyTorch C++ 库  
3. **OpenCV** - 计算机视觉库  
4. **Open3D** - 3D 数据处理库  

## 环境配置

### LibTorch 设置

项目依赖 LibTorch 进行深度学习推理。你可以选择以下方式之一：

#### 选项 1：设置环境变量（推荐）  
```bash
export LIBTORCH_PATH=/path/to/your/libtorch
```

#### 选项 2：安装到标准路径  
将 LibTorch 安装在以下标准路径之一：  
- `/usr/local/libtorch`  
- `/opt/libtorch`  
- `~/tools/libtorch`  
- `${PROJECT_ROOT}/libtorch`  

#### 选项 3：下载 LibTorch  
```bash
# 下载 LibTorch（CPU 版）
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.12.1%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-1.12.1+cpu.zip
export LIBTORCH_PATH=$(pwd)/libtorch
```

## 编译说明

1. **克隆仓库**  
```bash
git clone https://github.com/your-username/FM-Fusion.git
cd FM-Fusion
```

2. **编译主库**  
```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
cd ..
```

3. **编译 ROS 包**  
```bash
cd catkin_ws
catkin_make -DCMAKE_BUILD_TYPE=Release
source devel/setup.bash
```

## 配置文件

所有配置文件均使用相对路径，应开箱即用：

- `config/online.yaml` - 在线建图配置  
- `config/scannet_offline.yaml` - 离线建图配置  

## 数据准备

1. **创建数据目录结构**  
```bash
mkdir -p data/ScanNet/scans
mkdir -p output
```

2. **下载示例数据（可选）**  
将 ScanNet 数据放置到 `data/ScanNet/scans/` 目录下  

## 系统运行

### 在线建图  
```bash
cd catkin_ws
source devel/setup.bash

# 启动在线建图节点
rosrun sgloop_ros OnlineMappingNode \
    _cfg_file:=../config/online.yaml \
    _local_agent:=agent0 \
    _output_folder:=../output/online_test \
    _max_frames:=1000

# 在另一个终端播放 bag 文件
rosbag play your_data.bag --clock
```

### 离线建图  
```bash
cd catkin_ws
source devel/setup.bash

# 运行离线建图
rosrun sgloop_ros MappingNode \
    _cfg_file:=../config/scannet_offline.yaml \
    _active_sequence_dir:=../data/ScanNet/scans/your_scene \
    _local_agent:=agent0 \
    _output_folder:=../output/offline_test \
    _max_frames:=1000
```

## 常见问题排查

### 找不到 LibTorch  
如果遇到 LibTorch 相关错误：  
1. 确认已安装 LibTorch  
2. 设置 `LIBTORCH_PATH` 环境变量  
3. 重新编译项目  

### 找不到 ROS 包  
如果 ROS 找不到 sgloop_ros 包：  
```bash
cd catkin_ws
source devel/setup.bash
```

### 权限错误  
确保脚本有可执行权限：  
```bash
chmod +x scripts/*.sh
chmod +x scripts/*.py
```

## 开发说明

### VSCode 配置  
`.vscode/launch.json` 文件已配置为使用相对路径，设置环境后可直接使用。

### 添加新配置  
添加新的配置文件或脚本时，请使用相对路径或者 ROS 命令 `$(find sgloop_ros)`，以保证可移植性。

## 备注

- 配置文件中的所有路径均已相对于项目根目录  
- 脚本自动检测项目根目录  
- 系统设计保证无论仓库克隆至何处均能正常工作  
- 运行任何 ROS 命令前，请确保已执行 ROS 环境设置文件（source）