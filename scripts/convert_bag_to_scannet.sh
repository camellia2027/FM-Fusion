#!/bin/bash

# 转换bag文件到ScanNet格式的便捷脚本

# 获取脚本目录并构建相对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 设置路径
BAG_PATH="$PROJECT_ROOT/data/ScanNet/newdata.bag"
OUTPUT_DIR="$PROJECT_ROOT/data/ScanNet/scans"
SCENE_NAME="scene_from_online_bag"

echo "=== Converting bag file to ScanNet format ==="
echo "Input bag: $BAG_PATH"
echo "Output dir: $OUTPUT_DIR"
echo "Scene name: $SCENE_NAME"
echo

# 检查bag文件是否存在
if [ ! -f "$BAG_PATH" ]; then
    echo "Error: Bag file not found: $BAG_PATH"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 运行转换脚本
cd "$PROJECT_ROOT"
python3 scripts/bag_to_scannet_format.py "$BAG_PATH" "$OUTPUT_DIR" --scene_name "$SCENE_NAME"

echo
echo "=== Conversion completed ==="
echo "Check the output in: $OUTPUT_DIR/$SCENE_NAME"
echo
echo "Directory structure:"
ls -la "$OUTPUT_DIR/$SCENE_NAME"
echo
echo "Sample files:"
echo "Color images:"
ls "$OUTPUT_DIR/$SCENE_NAME/color" | head -5
echo "Depth images:"
ls "$OUTPUT_DIR/$SCENE_NAME/depth" | head -5
echo "Prediction files:"
ls "$OUTPUT_DIR/$SCENE_NAME/prediction_no_augment" | head -10
