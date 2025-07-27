#!/usr/bin/env python3

import cv2
import numpy as np
import sys

def analyze_depth_image(depth_path):
    """分析深度图像质量"""
    
    print(f"=== 分析深度图像: {depth_path} ===")
    
    # 读取深度图像
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        print(f"无法读取深度图像: {depth_path}")
        return
    
    print(f"深度图像尺寸: {depth.shape}")
    print(f"深度图像数据类型: {depth.dtype}")
    
    # 分析深度值分布
    valid_depth = depth[depth > 0]
    if len(valid_depth) == 0:
        print("❌ 没有有效的深度值！")
        return
    
    print(f"有效深度像素数量: {len(valid_depth)} / {depth.size} ({len(valid_depth)/depth.size*100:.1f}%)")
    print(f"深度值范围: {valid_depth.min()} - {valid_depth.max()}")
    print(f"深度值均值: {valid_depth.mean():.1f}")
    print(f"深度值中位数: {np.median(valid_depth):.1f}")
    
    # 分析深度分布
    depth_ranges = [
        (0, 500, "0-0.5m"),
        (500, 1000, "0.5-1m"),
        (1000, 2000, "1-2m"),
        (2000, 3000, "2-3m"),
        (3000, 4000, "3-4m"),
        (4000, float('inf'), ">4m")
    ]
    
    print("\n深度分布:")
    for min_d, max_d, label in depth_ranges:
        count = np.sum((valid_depth >= min_d) & (valid_depth < max_d))
        percentage = count / len(valid_depth) * 100
        print(f"  {label}: {count} 像素 ({percentage:.1f}%)")

def main():
    if len(sys.argv) != 2:
        print("用法: python3 analyze_depth_quality.py <depth_path>")
        return
    
    depth_path = sys.argv[1]
    analyze_depth_image(depth_path)

if __name__ == '__main__':
    main()
