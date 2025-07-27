#!/usr/bin/env python3

import rospy
import rosbag
import cv2
import numpy as np
import json
import os
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from std_msgs.msg import Header

class BagReader:
    def __init__(self):
        self.bag_file = 'data/ScanNet/newdata.bag'
        self.bridge = CvBridge()
        self.output_dir = 'data/ScanNet/extracted_data'
        
        # 创建输出目录
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            os.makedirs(os.path.join(self.output_dir, 'rgb'))
            os.makedirs(os.path.join(self.output_dir, 'depth'))
            os.makedirs(os.path.join(self.output_dir, 'mask'))
            os.makedirs(os.path.join(self.output_dir, 'json'))
            os.makedirs(os.path.join(self.output_dir, 'poses'))
        
        print(f"BagReader initialized for: {self.bag_file}")
        print(f"Output directory: {self.output_dir}")
    
    def read_bag_data(self):
        """读取bag文件并分析内容"""
        try:
            bag = rosbag.Bag(self.bag_file, 'r')
            
            frame_count = 0
            poses = []
            
            print("Starting to read bag data...")
            print("=" * 50)
            
            for topic, msg, t in bag.read_messages():
                if topic == '/sync/output':
                    print(f"\nFrame {frame_count}:")
                    print(f"  Timestamp: {t}")
                    print(f"  Message type: {type(msg)}")
                    
                    # 分析消息结构
                    self.analyze_message_structure(msg, frame_count)
                    
                    # 提取数据
                    success = self.extract_frame_data(msg, frame_count, t)
                    
                    if success:
                        frame_count += 1
                    
                    # 只处理前几帧来分析结构
                    if frame_count >= 5:
                        print(f"\n已分析前{frame_count}帧，继续处理所有帧...")
                        break
            
            # 处理所有帧
            frame_count = 0
            for topic, msg, t in bag.read_messages():
                if topic == '/sync/output':
                    success = self.extract_frame_data(msg, frame_count, t)
                    if success:
                        frame_count += 1
                        if frame_count % 5 == 0:
                            print(f"已处理 {frame_count} 帧")
            
            bag.close()
            print(f"\n完成！总共处理了 {frame_count} 帧数据")
            print(f"数据保存在: {self.output_dir}")
            
        except Exception as e:
            print(f"读取bag文件时出错: {e}")
            import traceback
            traceback.print_exc()
    
    def analyze_message_structure(self, msg, frame_id):
        """分析消息结构"""
        print(f"  消息字段:")
        
        # 检查消息的所有属性
        for attr in dir(msg):
            if not attr.startswith('_'):
                try:
                    value = getattr(msg, attr)
                    if hasattr(value, 'header') and hasattr(value.header, 'stamp'):
                        print(f"    {attr}: {type(value)} (有header)")
                    elif hasattr(value, 'x') and hasattr(value, 'y'):
                        print(f"    {attr}: {type(value)} (位置/点)")
                    elif isinstance(value, str):
                        print(f"    {attr}: string (长度: {len(value)})")
                    else:
                        print(f"    {attr}: {type(value)}")
                except:
                    print(f"    {attr}: 无法访问")
    
    def extract_frame_data(self, msg, frame_id, timestamp):
        """提取帧数据"""
        try:
            # 提取RGB图像
            if hasattr(msg, 'rgb'):
                rgb_success = self.save_image(msg.rgb, frame_id, 'rgb')
            elif hasattr(msg, 'color'):
                rgb_success = self.save_image(msg.color, frame_id, 'rgb')
            else:
                print(f"    警告: 帧{frame_id}没有找到RGB图像")
                rgb_success = False
            
            # 提取深度图像
            if hasattr(msg, 'depth'):
                depth_success = self.save_image(msg.depth, frame_id, 'depth')
            else:
                print(f"    警告: 帧{frame_id}没有找到深度图像")
                depth_success = False
            
            # 提取掩码图像
            if hasattr(msg, 'mask'):
                mask_success = self.save_image(msg.mask, frame_id, 'mask')
            else:
                print(f"    警告: 帧{frame_id}没有找到掩码图像")
                mask_success = False
            
            # 提取位姿
            if hasattr(msg, 'pose'):
                pose_success = self.save_pose(msg.pose, frame_id, timestamp)
            else:
                print(f"    警告: 帧{frame_id}没有找到位姿")
                pose_success = False
            
            # 提取JSON数据
            if hasattr(msg, 'json'):
                json_success = self.save_json(msg.json, frame_id)
            elif hasattr(msg, 'detection_json'):
                json_success = self.save_json(msg.detection_json, frame_id)
            else:
                print(f"    警告: 帧{frame_id}没有找到JSON数据")
                json_success = False
            
            return rgb_success or depth_success  # 至少要有图像数据
            
        except Exception as e:
            print(f"    错误: 提取帧{frame_id}数据失败: {e}")
            return False
    
    def save_image(self, img_msg, frame_id, img_type):
        """保存图像"""
        try:
            if img_type == 'rgb':
                cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            elif img_type == 'depth':
                cv_image = self.bridge.imgmsg_to_cv2(img_msg, "passthrough")
            else:  # mask
                cv_image = self.bridge.imgmsg_to_cv2(img_msg, "mono8")
            
            filename = os.path.join(self.output_dir, img_type, f"frame-{frame_id:06d}.png")
            cv2.imwrite(filename, cv_image)
            
            if frame_id < 5:  # 只为前几帧打印详细信息
                print(f"    保存{img_type}图像: {cv_image.shape}, {filename}")
            
            return True
            
        except Exception as e:
            print(f"    错误: 保存{img_type}图像失败: {e}")
            return False
    
    def save_pose(self, pose_msg, frame_id, timestamp):
        """保存位姿"""
        try:
            pose_data = {
                'frame_id': frame_id,
                'timestamp': timestamp.to_sec(),
                'position': {
                    'x': pose_msg.position.x,
                    'y': pose_msg.position.y,
                    'z': pose_msg.position.z
                },
                'orientation': {
                    'x': pose_msg.orientation.x,
                    'y': pose_msg.orientation.y,
                    'z': pose_msg.orientation.z,
                    'w': pose_msg.orientation.w
                }
            }
            
            filename = os.path.join(self.output_dir, 'poses', f"frame-{frame_id:06d}.json")
            with open(filename, 'w') as f:
                json.dump(pose_data, f, indent=2)
            
            if frame_id < 5:  # 只为前几帧打印详细信息
                print(f"    保存位姿: {filename}")
                print(f"      位置: ({pose_msg.position.x:.3f}, {pose_msg.position.y:.3f}, {pose_msg.position.z:.3f})")
            
            return True
            
        except Exception as e:
            print(f"    错误: 保存位姿失败: {e}")
            return False
    
    def save_json(self, json_str, frame_id):
        """保存JSON数据"""
        try:
            filename = os.path.join(self.output_dir, 'json', f"frame-{frame_id:06d}.json")
            
            # 如果是字符串，直接保存
            if isinstance(json_str, str):
                with open(filename, 'w') as f:
                    f.write(json_str)
            else:
                # 如果是其他格式，转换为JSON
                with open(filename, 'w') as f:
                    json.dump(json_str, f, indent=2)
            
            if frame_id < 5:  # 只为前几帧打印详细信息
                print(f"    保存JSON: {filename}")
                if isinstance(json_str, str) and len(json_str) > 0:
                    try:
                        parsed = json.loads(json_str)
                        print(f"      JSON包含: {list(parsed.keys()) if isinstance(parsed, dict) else 'array'}")
                    except:
                        print(f"      JSON长度: {len(json_str)}")
            
            return True
            
        except Exception as e:
            print(f"    错误: 保存JSON失败: {e}")
            return False

def main():
    reader = BagReader()
    reader.read_bag_data()

if __name__ == '__main__':
    main()
