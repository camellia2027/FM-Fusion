#!/usr/bin/env python3

import rospy
import rosbag
import cv2
import numpy as np
import json
from cv_bridge import CvBridge

def check_multiple_frames():
    """检查多帧数据的质量"""
    
    # Get the script directory and construct relative path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    bag_path = os.path.join(project_root, 'data', 'ScanNet', 'newdata.bag')
    bridge = CvBridge()
    
    print("=== 检查多帧掩码数据质量 ===")
    
    frame_count = 0
    with rosbag.Bag(bag_path, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=['/sync/output']):
            if frame_count >= 5:  # 只检查前5帧
                break
                
            print(f"\n--- 帧 {frame_count} ---")
            
            # 解析JSON
            json_data = json.loads(msg.json)
            detection_ids = [det['value'] for det in json_data['detections'] if det['value'] != 0]
            print(f"非背景检测ID: {detection_ids}")
            
            # 转换掩码图像
            mask_cv = bridge.imgmsg_to_cv2(msg.mask, desired_encoding='passthrough')
            b_channel = mask_cv[:, :, 0]  # B通道
            
            # 统计每个检测ID的像素数
            print(f"各ID像素统计:")
            total_instance_pixels = 0
            for det_id in detection_ids:
                pixel_count = np.sum(b_channel == det_id)
                print(f"  ID {det_id}: {pixel_count} 像素")
                total_instance_pixels += pixel_count
            
            background_pixels = np.sum(b_channel == 0)
            print(f"  背景 (ID 0): {background_pixels} 像素")
            print(f"  总实例像素: {total_instance_pixels}")
            print(f"  实例像素占比: {100.0 * total_instance_pixels / (640*480):.2f}%")
            
            # 检查是否有大实例
            large_instances = [det_id for det_id in detection_ids if np.sum(b_channel == det_id) >= 100]
            if large_instances:
                print(f"  大实例 (>=100像素): {large_instances}")
            else:
                print(f"  无大实例 (所有实例 <100像素)")
            
            frame_count += 1

if __name__ == '__main__':
    check_multiple_frames()
