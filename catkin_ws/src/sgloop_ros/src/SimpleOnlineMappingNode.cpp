#include <iostream>
#include <memory>
#include <vector>
#include <fstream>

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <geometry_msgs/Pose.h>
#include <std_msgs/String.h>
#include <std_msgs/Header.h>
#include "sgloop_ros/SyncedFrame.h"

class SimpleOnlineMappingNode
{
private:
    ros::NodeHandle nh_;
    ros::NodeHandle nh_private_;
    
    // Simple subscriber for synchronized data
    ros::Subscriber synced_frame_sub_;
    
    // Parameters
    std::string LOCAL_AGENT_;
    std::string output_folder_;
    int frame_gap_;
    int max_frames_;
    bool debug_;
    int frame_count_;
    int processed_frame_count_;

public:
    SimpleOnlineMappingNode(ros::NodeHandle& nh, ros::NodeHandle& nh_private) 
        : nh_(nh), nh_private_(nh_private), 
          frame_count_(0), processed_frame_count_(0)
    {
        // Load parameters
        nh_private_.getParam("local_agent", LOCAL_AGENT_);
        frame_gap_ = nh_private_.param("frame_gap", 1);
        output_folder_ = nh_private_.param("output_folder", std::string(""));
        max_frames_ = nh_private_.param("max_frames", 5000);
        debug_ = nh_private_.param("debug", false);
        
        ROS_WARN("SimpleOnlineMappingNode started");
        
        if(output_folder_.size() > 0) {
            ROS_INFO("Output folder: %s", output_folder_.c_str());
        }
        
        // Initialize subscriber for synchronized data
        synced_frame_sub_ = nh_.subscribe("/synced_frame", 1, &SimpleOnlineMappingNode::syncedFrameCallback, this);
        
        ROS_INFO("SimpleOnlineMappingNode initialized. Waiting for synchronized messages...");
    }
    
    ~SimpleOnlineMappingNode()
    {
        ROS_INFO("SimpleOnlineMappingNode shutting down");
    }

private:
    void syncedFrameCallback(const sgloop_ros::SyncedFrame::ConstPtr& msg)
    {
        frame_count_++;
        
        // Apply frame gap
        if ((frame_count_ - 1) % frame_gap_ != 0) {
            return;
        }
        
        if (processed_frame_count_ >= max_frames_) {
            ROS_WARN("Reached maximum frames (%d), stopping processing", max_frames_);
            return;
        }
        
        ROS_INFO("Processing frame %d (total received: %d)...", processed_frame_count_, frame_count_);
        
        try {
            processFrame(msg);
            processed_frame_count_++;
            ROS_INFO("Successfully processed frame %d", processed_frame_count_);
        } catch (const std::exception& e) {
            ROS_ERROR("Error processing frame %d: %s", processed_frame_count_, e.what());
        }
    }
    
    void processFrame(const sgloop_ros::SyncedFrame::ConstPtr& msg)
    {
        // Simple processing - just log the message info
        ROS_INFO("Frame info:");
        ROS_INFO("  Header: seq=%d, stamp=%f, frame_id=%s", 
                 msg->header.seq, msg->header.stamp.toSec(), msg->header.frame_id.c_str());
        ROS_INFO("  RGB: %dx%d, encoding=%s", msg->rgb.width, msg->rgb.height, msg->rgb.encoding.c_str());
        ROS_INFO("  Depth: %dx%d, encoding=%s", msg->depth.width, msg->depth.height, msg->depth.encoding.c_str());
        ROS_INFO("  Mask: %dx%d, encoding=%s", msg->mask.width, msg->mask.height, msg->mask.encoding.c_str());
        ROS_INFO("  Pose: pos=(%.3f,%.3f,%.3f)", msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
        ROS_INFO("  JSON length: %zu characters", msg->json.length());
        
        // Simple JSON parsing test
        if (!msg->json.empty()) {
            size_t start = msg->json.find('[');
            size_t end = msg->json.find(']');
            if (start != std::string::npos && end != std::string::npos) {
                ROS_INFO("  JSON appears to be valid array format");
                
                // Count objects in JSON
                size_t count = 0;
                size_t pos = 0;
                while ((pos = msg->json.find("\"value\":", pos)) != std::string::npos) {
                    count++;
                    pos++;
                }
                ROS_INFO("  Found %zu objects in JSON", count);
            } else {
                ROS_WARN("  JSON format may be invalid");
            }
        }
    }

public:
    void finalize()
    {
        ROS_WARN("Finalizing with %d processed frames", processed_frame_count_);
        
        // Save simple summary
        if (!output_folder_.empty()) {
            std::string summary_file = output_folder_ + "/processing_summary.txt";
            std::ofstream out_file(summary_file);
            if (out_file.is_open()) {
                out_file << "Simple Online Mapping Summary\n";
                out_file << "============================\n";
                out_file << "Total frames received: " << frame_count_ << "\n";
                out_file << "Frames processed: " << processed_frame_count_ << "\n";
                out_file << "Frame gap: " << frame_gap_ << "\n";
                out_file << "Max frames: " << max_frames_ << "\n";
                out_file.close();
                ROS_INFO("Summary saved to: %s", summary_file.c_str());
            }
        }
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "SimpleOnlineMappingNode");
    ros::NodeHandle nh;
    ros::NodeHandle nh_private("~");
    
    SimpleOnlineMappingNode node(nh, nh_private);
    
    ros::spin();
    
    // Finalize when shutting down
    node.finalize();
    
    return 0;
}
