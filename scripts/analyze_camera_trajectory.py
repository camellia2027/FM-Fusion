#!/usr/bin/env python3

import numpy as np
import os
import sys
import glob

def analyze_trajectory(pose_dir):
    """分析相机轨迹"""
    
    print(f"=== 分析相机轨迹: {pose_dir} ===")
    
    # 读取所有位姿文件
    pose_files = sorted(glob.glob(os.path.join(pose_dir, "*.txt")))
    if not pose_files:
        print("❌ 没有找到位姿文件！")
        return
    
    print(f"位姿文件数量: {len(pose_files)}")
    
    positions = []
    rotations = []
    
    for pose_file in pose_files:
        try:
            transform = np.loadtxt(pose_file)
            if transform.shape == (4, 4):
                position = transform[:3, 3]
                rotation = transform[:3, :3]
                positions.append(position)
                rotations.append(rotation)
        except Exception as e:
            print(f"读取位姿文件失败: {pose_file}, 错误: {e}")
    
    if not positions:
        print("❌ 没有有效的位姿数据！")
        return
    
    positions = np.array(positions)
    
    print(f"有效位姿数量: {len(positions)}")
    
    # 分析位置变化
    print(f"\n=== 位置分析 ===")
    print(f"X范围: {positions[:, 0].min():.3f} - {positions[:, 0].max():.3f} (变化: {positions[:, 0].max() - positions[:, 0].min():.3f}m)")
    print(f"Y范围: {positions[:, 1].min():.3f} - {positions[:, 1].max():.3f} (变化: {positions[:, 1].max() - positions[:, 1].min():.3f}m)")
    print(f"Z范围: {positions[:, 2].min():.3f} - {positions[:, 2].max():.3f} (变化: {positions[:, 2].max() - positions[:, 2].min():.3f}m)")
    
    # 分析运动距离
    if len(positions) > 1:
        distances = []
        for i in range(1, len(positions)):
            dist = np.linalg.norm(positions[i] - positions[i-1])
            distances.append(dist)
        
        distances = np.array(distances)
        total_distance = np.sum(distances)
        
        print(f"\n=== 运动分析 ===")
        print(f"总运动距离: {total_distance:.3f}m")
        print(f"平均帧间距离: {distances.mean():.3f}m")
        print(f"最大帧间距离: {distances.max():.3f}m")
        print(f"最小帧间距离: {distances.min():.3f}m")
        
        # 分析运动速度
        static_frames = np.sum(distances < 0.01)  # 小于1cm认为是静止
        slow_frames = np.sum((distances >= 0.01) & (distances < 0.05))  # 1-5cm
        normal_frames = np.sum((distances >= 0.05) & (distances < 0.2))  # 5-20cm
        fast_frames = np.sum(distances >= 0.2)  # 大于20cm
        
        print(f"\n=== 运动速度分布 ===")
        print(f"静止帧 (<1cm): {static_frames} ({static_frames/len(distances)*100:.1f}%)")
        print(f"慢速帧 (1-5cm): {slow_frames} ({slow_frames/len(distances)*100:.1f}%)")
        print(f"正常帧 (5-20cm): {normal_frames} ({normal_frames/len(distances)*100:.1f}%)")
        print(f"快速帧 (>20cm): {fast_frames} ({fast_frames/len(distances)*100:.1f}%)")

def main():
    if len(sys.argv) != 2:
        print("用法: python3 analyze_camera_trajectory.py <pose_dir>")
        return
    
    pose_dir = sys.argv[1]
    analyze_trajectory(pose_dir)

if __name__ == '__main__':
    main()
