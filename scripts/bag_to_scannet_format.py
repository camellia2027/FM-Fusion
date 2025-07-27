#!/usr/bin/env python3

import rospy
import rosbag
import cv2
import numpy as np
import json
import os
import argparse
from cv_bridge import CvBridge
from pathlib import Path

class BagToScanNetConverter:
    def __init__(self, bag_path, output_dir, scene_name="scene_from_bag"):
        self.bag_path = bag_path
        self.output_dir = Path(output_dir)
        self.scene_name = scene_name
        self.bridge = CvBridge()
        
        # 创建输出目录结构
        self.scene_dir = self.output_dir / scene_name
        self.color_dir = self.scene_dir / "color"
        self.depth_dir = self.scene_dir / "depth"
        self.prediction_dir = self.scene_dir / "prediction_no_augment"
        self.pose_dir = self.scene_dir / "pose"

        # 创建所有必要的目录
        for dir_path in [self.color_dir, self.depth_dir, self.prediction_dir, self.pose_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        print(f"Created output directories in: {self.scene_dir}")
    
    def convert_bag(self):
        """转换bag文件到ScanNet格式"""
        
        print(f"Converting bag file: {self.bag_path}")
        print(f"Output directory: {self.scene_dir}")
        
        frame_count = 0
        
        with rosbag.Bag(self.bag_path, 'r') as bag:
            for topic, msg, t in bag.read_messages(topics=['/sync/output']):
                print(f"\n--- Processing frame {frame_count} ---")
                
                # 生成帧文件名（ScanNet格式：frame-XXXXXX）
                frame_name = f"frame-{frame_count:06d}"
                
                try:
                    # 1. 保存彩色图像
                    self.save_color_image(msg, frame_name)
                    
                    # 2. 保存深度图像
                    self.save_depth_image(msg, frame_name)
                    
                    # 3. 保存检测标签和掩码
                    self.save_detection_data(msg, frame_name)

                    # 4. 保存位姿数据
                    self.save_pose_data(msg, frame_name)

                    print(f"Successfully processed frame {frame_count}")
                    frame_count += 1
                    
                except Exception as e:
                    print(f"Error processing frame {frame_count}: {e}")
                    continue
        
        print(f"\nConversion completed! Processed {frame_count} frames.")
        print(f"Output saved to: {self.scene_dir}")
    
    def save_color_image(self, msg, frame_name):
        """保存彩色图像为JPG格式"""
        try:
            # 转换ROS图像消息到OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(msg.rgb, desired_encoding='bgr8')
            
            # 保存为JPG格式
            color_path = self.color_dir / f"{frame_name}.jpg"
            cv2.imwrite(str(color_path), cv_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            print(f"  Saved color image: {color_path}")
            
        except Exception as e:
            print(f"  Error saving color image: {e}")
            raise
    
    def save_depth_image(self, msg, frame_name):
        """保存深度图像为PNG格式"""
        try:
            # 转换ROS深度图像消息到OpenCV格式
            cv_depth = self.bridge.imgmsg_to_cv2(msg.depth, desired_encoding='passthrough')
            
            # ScanNet深度格式：16位PNG，单位为毫米
            # 假设输入深度单位为米，转换为毫米
            if cv_depth.dtype == np.float32 or cv_depth.dtype == np.float64:
                # 浮点深度（米）转换为16位整数（毫米）
                depth_mm = (cv_depth * 1000).astype(np.uint16)
            else:
                # 已经是整数格式，假设单位正确
                depth_mm = cv_depth.astype(np.uint16)
            
            # 保存为PNG格式
            depth_path = self.depth_dir / f"{frame_name}.png"
            cv2.imwrite(str(depth_path), depth_mm)
            
            print(f"  Saved depth image: {depth_path}")
            
        except Exception as e:
            print(f"  Error saving depth image: {e}")
            raise
    
    def save_detection_data(self, msg, frame_name):
        """保存检测数据（JSON标签和掩码图像）"""
        try:
            # 解析JSON检测数据
            json_data = json.loads(msg.json)
            detections = json_data.get('detections', [])
            
            # 转换掩码图像
            mask_cv = self.bridge.imgmsg_to_cv2(msg.mask, desired_encoding='passthrough')

            # 提取ID映射（使用BGR图像的第一个通道）
            if len(mask_cv.shape) == 3:
                id_mask = mask_cv[:, :, 0]  # B通道作为ID映射
            else:
                id_mask = mask_cv

            # 创建正确的ID掩码：只保留JSON中存在的检测ID
            valid_ids = set([det.get('value', 0) for det in detections])
            corrected_mask = np.zeros_like(id_mask, dtype=np.uint8)

            for valid_id in valid_ids:
                if valid_id in valid_ids:
                    corrected_mask[id_mask == valid_id] = valid_id

            id_mask = corrected_mask
            
            # 转换为ScanNet格式的标签数据
            scannet_labels = self.convert_to_scannet_format(detections, id_mask)
            
            # 保存JSON标签文件
            label_path = self.prediction_dir / f"{frame_name}_label.json"
            with open(label_path, 'w') as f:
                json.dump(scannet_labels, f)
            
            print(f"  Saved label file: {label_path}")
            
            # 保存掩码图像（8位灰度PNG）
            mask_path = self.prediction_dir / f"{frame_name}_mask.png"
            cv2.imwrite(str(mask_path), id_mask.astype(np.uint8))
            
            print(f"  Saved mask image: {mask_path}")
            
        except Exception as e:
            print(f"  Error saving detection data: {e}")
            raise
    
    def convert_to_scannet_format(self, detections, id_mask):
        """将检测数据转换为ScanNet格式"""

        # 收集所有标签
        all_tags = []
        valid_detections = []

        for detection in detections:
            det_id = detection.get('value', 0)
            if det_id == 0:  # 跳过背景
                continue

            # 检查掩码中是否有对应的像素
            pixel_count = np.sum(id_mask == det_id)
            if pixel_count == 0:
                continue

            # 提取标签信息 - 修复：直接使用label和logit
            label = detection.get('label', '')
            logit = detection.get('logit', 0.0)
            box = detection.get('box', [0.0, 0.0, 0.0, 0.0])

            if not label:
                continue

            # 创建labels字典格式
            labels = {label: float(logit)}
                
            # 使用原始边界框（如果有的话），否则计算
            if len(box) == 4 and any(b != 0.0 for b in box):
                bbox = [float(b) for b in box]
            else:
                # 从掩码计算边界框
                mask_pixels = np.where(id_mask == det_id)
                if len(mask_pixels[0]) > 0:
                    y_min, y_max = mask_pixels[0].min(), mask_pixels[0].max()
                    x_min, x_max = mask_pixels[1].min(), mask_pixels[1].max()
                    bbox = [float(x_min), float(y_min), float(x_max), float(y_max)]
                else:
                    bbox = [0.0, 0.0, 0.0, 0.0]

            # 添加到有效检测列表
            valid_detections.append({
                "value": det_id,
                "labels": labels,
                "box": bbox
            })

            # 收集标签用于raw_tags
            all_tags.append(label)
        
        # 创建ScanNet格式的输出
        scannet_format = {
            "raw_tags": ". ".join(sorted(set(all_tags))),
            "tags": ". ".join(sorted(set(all_tags))),  # 简化版本，实际可能需要过滤
            "mask": [{"value": 0, "label": "background"}] + valid_detections
        }
        
        return scannet_format

    def save_pose_data(self, msg, frame_name):
        """保存位姿数据为TXT格式"""
        try:
            # 提取4x4变换矩阵
            transform_matrix = np.array(msg.transform_matrix).reshape(4, 4)

            # 保存为TXT格式（ScanNet格式）
            pose_path = self.pose_dir / f"{frame_name}.txt"
            np.savetxt(pose_path, transform_matrix, fmt='%.6f')

            print(f"  Saved pose file: {pose_path}")

        except Exception as e:
            print(f"  Error saving pose data: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Convert ROS bag to ScanNet format')
    parser.add_argument('bag_path', help='Path to the input bag file')
    parser.add_argument('output_dir', help='Output directory for ScanNet format data')
    parser.add_argument('--scene_name', default='scene_from_bag', 
                       help='Scene name (default: scene_from_bag)')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.bag_path):
        print(f"Error: Bag file not found: {args.bag_path}")
        return
    
    # 创建转换器并执行转换
    converter = BagToScanNetConverter(args.bag_path, args.output_dir, args.scene_name)
    converter.convert_bag()

if __name__ == '__main__':
    main()
