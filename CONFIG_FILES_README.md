# FM-Fusion Configuration Files

This document explains the different configuration files and their purposes.

## 📁 Configuration Files Overview

### `config/scannet.yaml` 
**Purpose**: Original ScanNet dataset configuration for **offline mapping**
- **DO NOT MODIFY** - This is for offline processing of ScanNet dataset
- Contains standard ScanNet camera parameters
- Used by original FM-Fusion offline pipeline

### `config/your_camera.yaml` ⭐
**Purpose**: Your actual camera configuration for **online mapping**
- Contains your real camera intrinsics (fx=902.16, fy=900.52, etc.)
- Scaled for 640x480 processing (fx=451.08, fy=600.34, etc.)
- Includes your distortion parameters (k1, k2, p1, p2)
- Includes IMU parameters for reference
- **Use this for online mapping with your hardware**

### `config/online_mapping.yaml`
**Purpose**: Alternative online mapping configuration
- Similar to your_camera.yaml but with different parameter organization
- Can be used as backup or for different camera setups

## 🎯 When to Use Which Config

### For Online Mapping (Real-time)
```bash
# Use your actual camera parameters
rosrun sgloop_ros OnlineMappingNode \
    _cfg_file:=/path/to/config/your_camera.yaml
```

### For Offline Mapping (ScanNet dataset)
```bash
# Use original ScanNet parameters
rosrun sgloop_ros MappingNode \
    _cfg_file:=/path/to/config/scannet.yaml
```

## 📊 Your Camera Parameters

### Original Camera (1280x720)
- fx: 902.1642456054688
- fy: 900.5171508789062  
- cx: 644.5207519531255
- cy: 360.89471435546875

### Scaled for Processing (640x480)
- fx: 451.08 (scaled by 0.5)
- fy: 600.34 (scaled by 0.667)
- cx: 322.26 (scaled by 0.5)
- cy: 240.60 (scaled by 0.667)

### Distortion Parameters
- k1: 9.2615504465028850e-02
- k2: -1.8082438825995681e-01
- p1: -6.5484100374765971e-04
- p2: -3.5829351558557421e-04

## 🔧 Image Processing Pipeline

1. **Your camera** captures 1280x720 images
2. **bag_to_synced_frame.py** resizes to 640x480
3. **FM-Fusion** processes with scaled intrinsics from your_camera.yaml
4. **Result** is geometrically correct 3D reconstruction

## 💡 Important Notes

- **Never modify scannet.yaml** - it's for offline ScanNet processing
- **Always use your_camera.yaml** for online mapping with your hardware
- **Image resizing** is handled automatically by bag_to_synced_frame.py
- **Intrinsic scaling** is pre-calculated in your_camera.yaml

## 🚀 Quick Start

For online mapping with your camera:
```bash
# Terminal 1: Start mapping
rosrun sgloop_ros OnlineMappingNode \
    _cfg_file:=/home/wuxin/Desktop/FM-Fusion/config/your_camera.yaml

# Terminal 2: Feed data (for testing with bag)
python3 bag_to_synced_frame.py data/to_fm.bag 1.0

# Terminal 3: Or use your real-time sync node
# (should publish SyncedFrame messages with 640x480 images)
```
