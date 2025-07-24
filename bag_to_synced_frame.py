#!/usr/bin/env python3

import rospy
import rosbag
from sgloop_ros.msg import SyncedFrame
from sensor_msgs.msg import Image
import sys
import cv2
import numpy as np
from cv_bridge import CvBridge

class BagToSyncedFrameConverter:
    def __init__(self):
        rospy.init_node('bag_to_synced_frame_converter')
        self.pub = rospy.Publisher('/synced_frame', SyncedFrame, queue_size=1)
        self.bridge = CvBridge()

        # Target image sizes to match scannet.yaml config
        self.target_rgb_width = 640
        self.target_rgb_height = 480
        self.target_depth_width = 640
        self.target_depth_height = 480

        rospy.loginfo("BagToSyncedFrameConverter initialized")
        rospy.loginfo("Target RGB size: %dx%d", self.target_rgb_width, self.target_rgb_height)
        rospy.loginfo("Target Depth size: %dx%d", self.target_depth_width, self.target_depth_height)

    def resize_image_msg(self, image_msg, target_width, target_height, interpolation=cv2.INTER_LINEAR):
        """Resize a ROS Image message to target dimensions"""
        try:
            # Convert ROS Image to OpenCV
            if image_msg.encoding == "rgb8":
                cv_image = self.bridge.imgmsg_to_cv2(image_msg, "rgb8")
            elif image_msg.encoding == "bgr8":
                cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
            elif image_msg.encoding == "16UC1":
                cv_image = self.bridge.imgmsg_to_cv2(image_msg, "16UC1")
            elif image_msg.encoding == "mono8":
                cv_image = self.bridge.imgmsg_to_cv2(image_msg, "mono8")
            else:
                cv_image = self.bridge.imgmsg_to_cv2(image_msg)

            # Resize image
            resized_image = cv2.resize(cv_image, (target_width, target_height), interpolation=interpolation)

            # Convert back to ROS Image
            if image_msg.encoding == "rgb8":
                resized_msg = self.bridge.cv2_to_imgmsg(resized_image, "rgb8")
            elif image_msg.encoding == "bgr8":
                resized_msg = self.bridge.cv2_to_imgmsg(resized_image, "bgr8")
            elif image_msg.encoding == "16UC1":
                resized_msg = self.bridge.cv2_to_imgmsg(resized_image, "16UC1")
            elif image_msg.encoding == "mono8":
                resized_msg = self.bridge.cv2_to_imgmsg(resized_image, "mono8")
            else:
                resized_msg = self.bridge.cv2_to_imgmsg(resized_image)

            # Copy header
            resized_msg.header = image_msg.header

            return resized_msg

        except Exception as e:
            rospy.logerr("Error resizing image: %s", str(e))
            return image_msg  # Return original if resize fails

    def convert_bag(self, bag_path, playback_rate=1.0):
        """Convert bag file to SyncedFrame messages"""
        
        try:
            bag = rosbag.Bag(bag_path, 'r')
            rospy.loginfo(f"Converting bag file: {bag_path}")
            
            message_count = 0
            start_time = None
            
            for topic, msg, t in bag.read_messages():
                if rospy.is_shutdown():
                    break
                    
                try:
                    # Create SyncedFrame message
                    synced_msg = SyncedFrame()

                    # Copy header and pose (no changes needed)
                    synced_msg.header = msg.header
                    synced_msg.pose = msg.pose
                    synced_msg.json = msg.json

                    # Resize images to match scannet.yaml config (640x480)
                    rospy.loginfo("Original sizes - RGB: %dx%d, Depth: %dx%d, Mask: %dx%d",
                                 msg.rgb.width, msg.rgb.height,
                                 msg.depth.width, msg.depth.height,
                                 msg.mask.width, msg.mask.height)

                    # Resize RGB image (1280x720 -> 640x480)
                    synced_msg.rgb = self.resize_image_msg(msg.rgb, self.target_rgb_width, self.target_rgb_height)

                    # Resize depth image (848x480 -> 640x480)
                    synced_msg.depth = self.resize_image_msg(msg.depth, self.target_depth_width, self.target_depth_height, cv2.INTER_NEAREST)

                    # Resize mask image (1280x720 -> 640x480) - use nearest neighbor to preserve mask values
                    synced_msg.mask = self.resize_image_msg(msg.mask, self.target_rgb_width, self.target_rgb_height, cv2.INTER_NEAREST)

                    rospy.loginfo("Resized to - RGB: %dx%d, Depth: %dx%d, Mask: %dx%d",
                                 synced_msg.rgb.width, synced_msg.rgb.height,
                                 synced_msg.depth.width, synced_msg.depth.height,
                                 synced_msg.mask.width, synced_msg.mask.height)
                    
                    message_count += 1
                    rospy.loginfo(f"Converting message {message_count}: {t}")
                    
                    # Publish converted message
                    self.pub.publish(synced_msg)
                    
                    # Sleep to simulate real-time playback
                    if playback_rate > 0:
                        rospy.sleep(1.0 / playback_rate)
                    
                except Exception as e:
                    rospy.logerr(f"Error converting message {message_count}: {e}")
                    
            bag.close()
            rospy.loginfo(f"Conversion completed! Processed {message_count} messages.")
            
        except Exception as e:
            rospy.logerr(f"Error reading bag file: {e}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 bag_to_synced_frame.py <bag_file_path> [playback_rate]")
        print("  bag_file_path: Path to the bag file")
        print("  playback_rate: Playback speed (default: 1.0, 0 = as fast as possible)")
        sys.exit(1)
        
    bag_path = sys.argv[1]
    playback_rate = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    
    converter = BagToSyncedFrameConverter()
    converter.convert_bag(bag_path, playback_rate)
