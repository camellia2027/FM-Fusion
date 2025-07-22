#!/usr/bin/env python3
"""
简单的PLY文件查看器
使用matplotlib显示点云的2D投影
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def read_ply_file(filename):
    """读取PLY文件"""
    points = []
    colors = []
    
    with open(filename, 'r') as f:
        # 跳过头部
        line = f.readline()
        while line.strip() != 'end_header':
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[-1])
            line = f.readline()
        
        # 读取点数据
        for i in range(vertex_count):
            line = f.readline().strip()
            if line:
                parts = line.split()
                if len(parts) >= 3:
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    points.append([x, y, z])
                    
                    # 如果有颜色信息
                    if len(parts) >= 6:
                        r, g, b = int(parts[3]), int(parts[4]), int(parts[5])
                        colors.append([r/255.0, g/255.0, b/255.0])
                    else:
                        colors.append([0.5, 0.5, 0.5])  # 默认灰色
    
    return np.array(points), np.array(colors)

def visualize_ply_2d(filename):
    """2D可视化PLY文件"""
    print(f"读取文件: {filename}")
    
    if not os.path.exists(filename):
        print(f"文件不存在: {filename}")
        return
    
    try:
        points, colors = read_ply_file(filename)
        print(f"读取到 {len(points)} 个点")
        
        if len(points) == 0:
            print("没有点数据!")
            return
        
        # 创建2D投影图
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'PLY文件可视化: {os.path.basename(filename)}')
        
        # XY投影 (俯视图)
        axes[0,0].scatter(points[:, 0], points[:, 1], c=colors, s=1, alpha=0.6)
        axes[0,0].set_title('XY投影 (俯视图)')
        axes[0,0].set_xlabel('X')
        axes[0,0].set_ylabel('Y')
        axes[0,0].axis('equal')
        
        # XZ投影 (侧视图)
        axes[0,1].scatter(points[:, 0], points[:, 2], c=colors, s=1, alpha=0.6)
        axes[0,1].set_title('XZ投影 (侧视图)')
        axes[0,1].set_xlabel('X')
        axes[0,1].set_ylabel('Z')
        axes[0,1].axis('equal')
        
        # YZ投影 (正视图)
        axes[1,0].scatter(points[:, 1], points[:, 2], c=colors, s=1, alpha=0.6)
        axes[1,0].set_title('YZ投影 (正视图)')
        axes[1,0].set_xlabel('Y')
        axes[1,0].set_ylabel('Z')
        axes[1,0].axis('equal')
        
        # 统计信息
        axes[1,1].text(0.1, 0.8, f'点数: {len(points):,}', transform=axes[1,1].transAxes)
        axes[1,1].text(0.1, 0.7, f'X范围: {points[:, 0].min():.2f} ~ {points[:, 0].max():.2f}', transform=axes[1,1].transAxes)
        axes[1,1].text(0.1, 0.6, f'Y范围: {points[:, 1].min():.2f} ~ {points[:, 1].max():.2f}', transform=axes[1,1].transAxes)
        axes[1,1].text(0.1, 0.5, f'Z范围: {points[:, 2].min():.2f} ~ {points[:, 2].max():.2f}', transform=axes[1,1].transAxes)
        axes[1,1].set_title('统计信息')
        axes[1,1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"读取文件时出错: {e}")

def visualize_ply_3d(filename):
    """3D可视化PLY文件"""
    print(f"3D可视化: {filename}")
    
    if not os.path.exists(filename):
        print(f"文件不存在: {filename}")
        return
    
    try:
        points, colors = read_ply_file(filename)
        print(f"读取到 {len(points)} 个点")
        
        if len(points) == 0:
            print("没有点数据!")
            return
        
        # 为了性能，如果点太多就采样
        if len(points) > 10000:
            indices = np.random.choice(len(points), 10000, replace=False)
            points = points[indices]
            colors = colors[indices]
            print(f"采样到 {len(points)} 个点")
        
        # 创建3D图
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                  c=colors, s=1, alpha=0.6)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'3D点云: {os.path.basename(filename)}')
        
        plt.show()
        
    except Exception as e:
        print(f"3D可视化时出错: {e}")

def main():
    result_dir = "data/ScanNet/output/online_mapping/scene0025_00"
    
    print("=== FM-Fusion 结果可视化 ===")
    print("1. 完整地图")
    print("2. 各个实例")
    print("3. 退出")
    
    while True:
        choice = input("\n请选择 (1-3): ").strip()
        
        if choice == '1':
            # 可视化完整地图
            complete_map = os.path.join(result_dir, "instance_map.ply")
            print("\n选择可视化模式:")
            print("1. 2D投影")
            print("2. 3D视图")
            mode = input("请选择 (1-2): ").strip()
            
            if mode == '1':
                visualize_ply_2d(complete_map)
            elif mode == '2':
                visualize_ply_3d(complete_map)
                
        elif choice == '2':
            # 可视化各个实例
            if not os.path.exists(result_dir):
                print(f"结果目录不存在: {result_dir}")
                continue
                
            ply_files = [f for f in os.listdir(result_dir) 
                        if f.endswith('.ply') and f != 'instance_map.ply']
            ply_files.sort()
            
            if not ply_files:
                print("没有找到实例文件!")
                continue
                
            print(f"\n找到 {len(ply_files)} 个实例:")
            for i, f in enumerate(ply_files):
                print(f"{i+1}. {f}")
            
            try:
                idx = int(input("请选择实例编号: ")) - 1
                if 0 <= idx < len(ply_files):
                    ply_file = os.path.join(result_dir, ply_files[idx])
                    print("\n选择可视化模式:")
                    print("1. 2D投影")
                    print("2. 3D视图")
                    mode = input("请选择 (1-2): ").strip()
                    
                    if mode == '1':
                        visualize_ply_2d(ply_file)
                    elif mode == '2':
                        visualize_ply_3d(ply_file)
                else:
                    print("无效的编号!")
            except ValueError:
                print("请输入有效的数字!")
                
        elif choice == '3':
            print("退出")
            break
        else:
            print("无效的选择!")

if __name__ == "__main__":
    main()
