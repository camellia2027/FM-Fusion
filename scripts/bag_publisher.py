#!/usr/bin/env python3

import rospy
import rosbag
import cv2
import numpy as np
import json
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from std_msgs.msg import Header
from sgloop_ros.msg import SyncedFrame

class BagPublisher:
    def __init__(self):
        rospy.init_node('bag_publisher', anonymous=True)
        
        # 参数
        self.bag_file = rospy.get_param('~bag_file', 'data/ScanNet/newdata.bag')
        self.publish_rate = rospy.get_param('~publish_rate', 10.0)
        self.use_bag_timestamps = rospy.get_param('~use_bag_timestamps', False)
        
        # 发布器
        self.synced_frame_pub = rospy.Publisher('/synced_frame', SyncedFrame, queue_size=10)
        
        # CV Bridge
        self.bridge = CvBridge()
        
        rospy.loginfo(f"BagPublisher initialized with bag file: {self.bag_file}")
        rospy.loginfo(f"Publishing rate: {self.publish_rate} Hz")
        rospy.loginfo(f"Use bag timestamps: {self.use_bag_timestamps}")
        
    def publish_bag_data(self):
        try:
            bag = rosbag.Bag(self.bag_file, 'r')
            
            rate = rospy.Rate(self.publish_rate)
            frame_count = 0
            start_time = rospy.Time.now()
            bag_start_time = None
            
            rospy.loginfo("Starting to publish bag data...")
            
            for topic, msg, t in bag.read_messages():
                if rospy.is_shutdown():
                    break
                    
                if topic == '/sync/output':
                    # 设置时间基准
                    if bag_start_time is None:
                        bag_start_time = t
                    
                    # 创建SyncedFrame消息
                    synced_frame = SyncedFrame()
                    
                    # 设置header
                    if self.use_bag_timestamps:
                        synced_frame.header.stamp = t
                    else:
                        elapsed = t - bag_start_time
                        synced_frame.header.stamp = start_time + elapsed
                    
                    synced_frame.header.seq = frame_count
                    synced_frame.header.frame_id = "camera"
                    
                    # 解析vins_estimator/SyncData消息
                    if self.parse_vins_sync_data(msg, synced_frame):
                        # 发布消息
                        self.synced_frame_pub.publish(synced_frame)
                        
                        if frame_count % 10 == 0:
                            rospy.loginfo(f"Published frame {frame_count}")
                        
                        frame_count += 1
                        
                        if not self.use_bag_timestamps:
                            rate.sleep()
                    else:
                        rospy.logwarn(f"Failed to parse message for frame {frame_count}")
            
            bag.close()
            rospy.loginfo(f"Finished publishing {frame_count} frames from bag file")
            
        except Exception as e:
            rospy.logerr(f"Error reading bag file: {e}")
    
    def parse_vins_sync_data(self, msg, synced_frame):
        """解析vins_estimator/SyncData消息"""
        try:
            # 直接复制图像数据
            if hasattr(msg, 'rgb'):
                synced_frame.rgb = msg.rgb
            else:
                rospy.logwarn("No RGB image found")
                return False
                
            if hasattr(msg, 'depth'):
                synced_frame.depth = msg.depth
            else:
                rospy.logwarn("No depth image found")
                return False
                
            if hasattr(msg, 'mask'):
                synced_frame.mask = msg.mask
            else:
                synced_frame.mask = self.create_empty_mask()
            
            # 设置transform_matrix
            if hasattr(msg, 'transform_matrix') and len(msg.transform_matrix) == 16:
                synced_frame.transform_matrix = list(msg.transform_matrix)
            else:
                synced_frame.transform_matrix = [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]  # 单位矩阵
                rospy.logwarn("No valid transform matrix found, using identity matrix")
            
            # 复制JSON数据
            if hasattr(msg, 'json'):
                synced_frame.json = msg.json
            else:
                synced_frame.json = self.create_empty_json()
            
            return True
            
        except Exception as e:
            rospy.logerr(f"Error parsing vins sync data: {e}")
            return False
    

    
    def create_empty_mask(self):
        """创建空掩码图像"""
        mask_image = np.zeros((480, 640), dtype=np.uint8)
        return self.bridge.cv2_to_imgmsg(mask_image, "mono8")
    

    
    def create_empty_json(self):
        """创建空的检测JSON"""
        data = {
            "header": {
                "stamp": {"sec": 0, "nsec": 0},
                "frame_id": "camera"
            },
            "detections": []
        }
        return json.dumps(data)

def main():
    try:
        publisher = BagPublisher()
        publisher.publish_bag_data()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
