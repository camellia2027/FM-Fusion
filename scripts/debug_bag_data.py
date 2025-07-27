#!/usr/bin/env python3

import rospy
import rosbag
import json
import numpy as np
from cv_bridge import CvBridge

def debug_bag_data():
    """调试bag文件中的数据内容"""
    
    bag_path = '/home/wuxin/Desktop/FM-Fusion/data/ScanNet/newdata.bag'
    bridge = CvBridge()
    
    print("=== 调试bag文件数据 ===")
    
    frame_count = 0
    with rosbag.Bag(bag_path, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=['/sync/output']):
            if frame_count >= 5:  # 只检查前5帧
                break
                
            print(f"\n--- 帧 {frame_count} ---")
            
            # 1. 检查JSON数据
            try:
                json_data = json.loads(msg.json)
                print(f"JSON keys: {json_data.keys()}")
                
                if 'detections' in json_data:
                    detections = json_data['detections']
                    print(f"检测数量: {len(detections)}")
                    
                    for i, det in enumerate(detections[:3]):  # 只显示前3个
                        print(f"  检测 {i}: {det}")
                else:
                    print("JSON中没有'detections'字段")
                    print(f"完整JSON: {json_data}")
                    
            except Exception as e:
                print(f"JSON解析错误: {e}")
                print(f"原始JSON字符串: {msg.json[:200]}...")
            
            # 2. 检查掩码数据
            try:
                mask_cv = bridge.imgmsg_to_cv2(msg.mask, desired_encoding='passthrough')
                print(f"掩码形状: {mask_cv.shape}")
                print(f"掩码数据类型: {mask_cv.dtype}")
                
                if len(mask_cv.shape) == 3:
                    b_channel = mask_cv[:, :, 0]
                    unique_vals = np.unique(b_channel)
                    print(f"B通道唯一值: {unique_vals[:10]}...")  # 只显示前10个
                else:
                    unique_vals = np.unique(mask_cv)
                    print(f"掩码唯一值: {unique_vals[:10]}...")
                    
            except Exception as e:
                print(f"掩码处理错误: {e}")
            
            # 3. 检查位姿数据
            try:
                transform_matrix = np.array(msg.transform_matrix).reshape(4, 4)
                print(f"位姿矩阵:\n{transform_matrix}")
            except Exception as e:
                print(f"位姿处理错误: {e}")
            
            frame_count += 1

if __name__ == '__main__':
    debug_bag_data()
