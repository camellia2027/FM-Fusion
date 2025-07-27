#!/usr/bin/env python3

"""
Real-time Data Publisher for Online Mapping

This script demonstrates how to publish real-time sensor data
to the /sync/output topic for online mapping.

Usage:
    python3 realtime_data_publisher.py
"""

import rospy
import cv2
import numpy as np
import json
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge
from sgloop_ros.msg import SyncedFrame

class RealtimeDataPublisher:
    def __init__(self):
        rospy.init_node('realtime_data_publisher', anonymous=True)
        
        # Publisher for synchronized frame data
        self.synced_pub = rospy.Publisher('/sync/output', SyncedFrame, queue_size=1)
        
        # CV Bridge for image conversion
        self.bridge = CvBridge()
        
        # Frame counter
        self.frame_count = 0
        
        rospy.loginfo("Real-time data publisher initialized")
    
    def create_sample_detection_json(self, frame_id):
        """Create sample detection JSON data"""
        detections = [
            {
                "value": 1,
                "label": "wall",
                "confidence": 0.95,
                "bbox": [100, 100, 200, 200]
            },
            {
                "value": 2, 
                "label": "chair",
                "confidence": 0.88,
                "bbox": [300, 150, 400, 300]
            }
        ]
        return json.dumps(detections)
    
    def create_sample_mask(self, height, width):
        """Create sample mask image with instance IDs"""
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Add some sample instances
        mask[100:200, 100:200] = 1  # Instance 1
        mask[150:300, 300:400] = 2  # Instance 2
        
        return mask
    
    def create_sample_transform(self, frame_id):
        """Create sample 4x4 transform matrix"""
        # Simple forward motion with slight rotation
        angle = frame_id * 0.01  # Small rotation per frame
        tx = frame_id * 0.05     # Forward motion
        
        transform = np.eye(4, dtype=np.float64)
        transform[0, 0] = np.cos(angle)
        transform[0, 2] = np.sin(angle)
        transform[2, 0] = -np.sin(angle)
        transform[2, 2] = np.cos(angle)
        transform[0, 3] = tx
        
        return transform.flatten().tolist()
    
    def publish_frame(self, rgb_image, depth_image):
        """
        Publish a synchronized frame with all required data
        
        Args:
            rgb_image: RGB image (numpy array, HxWx3, uint8)
            depth_image: Depth image (numpy array, HxW, uint16)
        """
        try:
            # Create header with current timestamp
            header = Header()
            header.stamp = rospy.Time.now()
            header.frame_id = f"frame_{self.frame_count:06d}"
            
            # Convert images to ROS messages
            rgb_msg = self.bridge.cv2_to_imgmsg(rgb_image, "bgr8")
            depth_msg = self.bridge.cv2_to_imgmsg(depth_image, "16UC1")
            
            # Create mask
            mask_array = self.create_sample_mask(rgb_image.shape[0], rgb_image.shape[1])
            mask_msg = self.bridge.cv2_to_imgmsg(mask_array, "mono8")
            
            # Create detection JSON
            detection_json = self.create_sample_detection_json(self.frame_count)
            
            # Create transform matrix
            transform_matrix = self.create_sample_transform(self.frame_count)
            
            # Create synchronized frame message
            synced_frame = SyncedFrame()
            synced_frame.header = header
            synced_frame.rgb = rgb_msg
            synced_frame.depth = depth_msg
            synced_frame.mask = mask_msg
            synced_frame.transform_matrix = transform_matrix
            synced_frame.json = detection_json
            
            # Publish the frame
            self.synced_pub.publish(synced_frame)
            
            rospy.loginfo(f"Published frame {self.frame_count}")
            self.frame_count += 1
            
        except Exception as e:
            rospy.logerr(f"Error publishing frame: {e}")
    
    def run_sample_stream(self):
        """Run a sample data stream for testing"""
        rate = rospy.Rate(2)  # 2 Hz for testing
        
        while not rospy.is_shutdown():
            # Create sample RGB image (640x480)
            rgb_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Add some patterns to make it more realistic
            cv2.rectangle(rgb_image, (100, 100), (200, 200), (0, 255, 0), -1)
            cv2.rectangle(rgb_image, (300, 150), (400, 300), (255, 0, 0), -1)
            
            # Create sample depth image
            depth_image = np.random.randint(500, 3000, (480, 640), dtype=np.uint16)
            
            # Publish the frame
            self.publish_frame(rgb_image, depth_image)
            
            rate.sleep()

def main():
    try:
        publisher = RealtimeDataPublisher()
        
        rospy.loginfo("Starting sample data stream...")
        rospy.loginfo("Publishing to /sync/output topic")
        rospy.loginfo("Use Ctrl+C to stop")
        
        publisher.run_sample_stream()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("Real-time data publisher stopped")

if __name__ == '__main__':
    main()
