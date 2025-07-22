#!/usr/bin/env python3
"""
FM-Fusion结果可视化脚本
使用Open3D可视化生成的语义实例地图
"""

import open3d as o3d
import numpy as np
import os
import argparse

def load_instance_info(info_file):
    """加载实例信息"""
    instances = {}
    if os.path.exists(info_file):
        with open(info_file, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split(';')
                if len(parts) >= 3:
                    instance_id = parts[0]
                    semantic_class = parts[1].split('(')[0]
                    point_count = parts[4] if len(parts) > 4 else "unknown"
                    instances[instance_id] = {
                        'class': semantic_class,
                        'points': point_count
                    }
    return instances

def visualize_complete_map(ply_file):
    """可视化完整的实例地图"""
    print(f"加载完整地图: {ply_file}")
    
    if not os.path.exists(ply_file):
        print(f"文件不存在: {ply_file}")
        return
    
    # 加载点云
    pcd = o3d.io.read_point_cloud(ply_file)
    print(f"点云包含 {len(pcd.points)} 个点")
    
    if len(pcd.points) == 0:
        print("点云为空!")
        return
    
    # 设置可视化
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="FM-Fusion 语义地图", width=1200, height=800)
    vis.add_geometry(pcd)
    
    # 设置渲染选项
    render_option = vis.get_render_option()
    render_option.point_size = 2.0
    render_option.background_color = np.array([0.1, 0.1, 0.1])
    
    print("可视化控制:")
    print("- 鼠标左键拖拽: 旋转")
    print("- 鼠标右键拖拽: 平移") 
    print("- 滚轮: 缩放")
    print("- 按 Q 或关闭窗口退出")
    
    vis.run()
    vis.destroy_window()

def visualize_individual_instances(result_dir):
    """可视化各个实例"""
    print(f"加载实例文件从: {result_dir}")
    
    # 加载实例信息
    info_file = os.path.join(result_dir, "instance_info.txt")
    instances = load_instance_info(info_file)
    
    # 找到所有.ply文件
    ply_files = []
    for file in os.listdir(result_dir):
        if file.endswith('.ply') and file != 'instance_map.ply':
            ply_files.append(file)
    
    ply_files.sort()
    
    if not ply_files:
        print("没有找到实例文件!")
        return
    
    print(f"找到 {len(ply_files)} 个实例文件")
    
    # 为每个实例分配颜色
    colors = [
        [1.0, 0.0, 0.0],  # 红色
        [0.0, 1.0, 0.0],  # 绿色
        [0.0, 0.0, 1.0],  # 蓝色
        [1.0, 1.0, 0.0],  # 黄色
        [1.0, 0.0, 1.0],  # 品红
        [0.0, 1.0, 1.0],  # 青色
        [1.0, 0.5, 0.0],  # 橙色
        [0.5, 0.0, 1.0],  # 紫色
        [0.0, 0.5, 0.0],  # 深绿
        [0.5, 0.5, 0.5],  # 灰色
        [1.0, 0.5, 0.5],  # 浅红
    ]
    
    # 加载所有实例
    all_pcds = []
    for i, ply_file in enumerate(ply_files):
        ply_path = os.path.join(result_dir, ply_file)
        pcd = o3d.io.read_point_cloud(ply_path)
        
        if len(pcd.points) > 0:
            # 设置颜色
            color = colors[i % len(colors)]
            pcd.paint_uniform_color(color)
            all_pcds.append(pcd)
            
            # 获取实例信息
            instance_id = ply_file.replace('.ply', '')
            instance_info = instances.get(instance_id, {'class': 'unknown', 'points': '0'})
            print(f"实例 {instance_id}: {instance_info['class']} ({instance_info['points']} 点)")
    
    if not all_pcds:
        print("没有有效的点云数据!")
        return
    
    # 可视化所有实例
    print(f"\n可视化 {len(all_pcds)} 个实例...")
    o3d.visualization.draw_geometries(
        all_pcds,
        window_name="FM-Fusion 语义实例",
        width=1200,
        height=800,
        point_show_normal=False
    )

def main():
    parser = argparse.ArgumentParser(description='可视化FM-Fusion结果')
    parser.add_argument('--result_dir', 
                       default='data/ScanNet/output/online_mapping/scene0025_00',
                       help='结果目录路径')
    parser.add_argument('--mode', 
                       choices=['complete', 'instances', 'both'],
                       default='both',
                       help='可视化模式')
    
    args = parser.parse_args()
    
    result_dir = args.result_dir
    
    if not os.path.exists(result_dir):
        print(f"结果目录不存在: {result_dir}")
        return
    
    print("=== FM-Fusion 结果可视化 ===")
    print(f"结果目录: {result_dir}")
    
    if args.mode in ['complete', 'both']:
        print("\n1. 可视化完整地图...")
        complete_map = os.path.join(result_dir, 'instance_map.ply')
        visualize_complete_map(complete_map)
    
    if args.mode in ['instances', 'both']:
        print("\n2. 可视化各个实例...")
        visualize_individual_instances(result_dir)

if __name__ == "__main__":
    main()
