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

#include <open3d/Open3D.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

// FM-Fusion headers with error checking
#include "tools/Utility.h"
#include "tools/IO.h"
#include "tools/TicToc.h"
#include "mapping/SemanticMapping.h"
#include "Visualization.h"

class SafeOnlineMappingNode
{
private:
    ros::NodeHandle nh_;
    ros::NodeHandle nh_private_;
    
    // Simple subscriber for synchronized data
    ros::Subscriber synced_frame_sub_;
    
    // FM-Fusion components
    fmfusion::Config *global_config_;
    fmfusion::SemanticMapping *semantic_mapping_;
    Visualization::Visualizer viz_;
    
    // Parameters
    std::string LOCAL_AGENT_;
    std::string output_folder_;
    std::string config_file_;
    int frame_gap_;
    int max_frames_;
    bool debug_;
    int frame_count_;
    int processed_frame_count_;
    bool initialized_;
    
    fmfusion::TicTocSequence tic_toc_seq_;

public:
    SafeOnlineMappingNode(ros::NodeHandle& nh, ros::NodeHandle& nh_private) 
        : nh_(nh), nh_private_(nh_private), viz_(nh, nh_private), 
          frame_count_(0), processed_frame_count_(0), initialized_(false),
          tic_toc_seq_("# Safe Online Mapping", 3),
          global_config_(nullptr), semantic_mapping_(nullptr)
    {
        ROS_WARN("SafeOnlineMappingNode starting initialization...");
        
        try {
            // Load parameters
            if (!nh_private_.getParam("cfg_file", config_file_)) {
                ROS_ERROR("Failed to get cfg_file parameter");
                return;
            }
            
            nh_private_.getParam("local_agent", LOCAL_AGENT_);
            frame_gap_ = nh_private_.param("frame_gap", 1);
            output_folder_ = nh_private_.param("output_folder", std::string(""));
            int o3d_verbose_level = nh_private_.param("o3d_verbose_level", 2);
            max_frames_ = nh_private_.param("max_frames", 100);
            debug_ = nh_private_.param("debug", false);
            
            ROS_INFO("Configuration:");
            ROS_INFO("  Config file: %s", config_file_.c_str());
            ROS_INFO("  Output folder: %s", output_folder_.c_str());
            ROS_INFO("  Max frames: %d", max_frames_);
            ROS_INFO("  Frame gap: %d", frame_gap_);
            
            // Check if config file exists
            if (!std::ifstream(config_file_).good()) {
                ROS_ERROR("Config file does not exist: %s", config_file_.c_str());
                return;
            }
            
            // Initialize FM-Fusion with error checking
            ROS_INFO("Initializing FM-Fusion...");
            global_config_ = fmfusion::utility::create_scene_graph_config(config_file_, true);
            if (!global_config_) {
                ROS_ERROR("Failed to create FM-Fusion config");
                return;
            }
            
            open3d::utility::SetVerbosityLevel((open3d::utility::VerbosityLevel)o3d_verbose_level);
            
            // Create output directory
            if(output_folder_.size() > 0 && !open3d::utility::filesystem::DirectoryExists(output_folder_)) {
                if (!open3d::utility::filesystem::MakeDirectory(output_folder_)) {
                    ROS_ERROR("Failed to create output directory: %s", output_folder_.c_str());
                    return;
                }
            }
            
            // Save config
            if (!output_folder_.empty()) {
                std::ofstream out_file(output_folder_ + "/config.txt");
                if (out_file.is_open()) {
                    out_file << fmfusion::utility::config_to_message(*global_config_);
                    out_file.close();
                    ROS_INFO("Config saved to: %s/config.txt", output_folder_.c_str());
                }
            }
            
            // Initialize semantic mapping
            ROS_INFO("Initializing semantic mapping...");
            semantic_mapping_ = new fmfusion::SemanticMapping(global_config_->mapping_cfg, global_config_->instance_cfg);
            if (!semantic_mapping_) {
                ROS_ERROR("Failed to create semantic mapping");
                return;
            }
            
            // Initialize subscriber
            synced_frame_sub_ = nh_.subscribe("/synced_frame", 1, &SafeOnlineMappingNode::syncedFrameCallback, this);
            
            initialized_ = true;
            ROS_WARN("SafeOnlineMappingNode initialized successfully!");
            
        } catch (const std::exception& e) {
            ROS_ERROR("Exception during initialization: %s", e.what());
            initialized_ = false;
        } catch (...) {
            ROS_ERROR("Unknown exception during initialization");
            initialized_ = false;
        }
    }
    
    ~SafeOnlineMappingNode()
    {
        ROS_INFO("SafeOnlineMappingNode shutting down...");
        if (semantic_mapping_) {
            delete semantic_mapping_;
            semantic_mapping_ = nullptr;
        }
        if (global_config_) {
            delete global_config_;
            global_config_ = nullptr;
        }
    }

private:
    void syncedFrameCallback(const sgloop_ros::SyncedFrame::ConstPtr& msg)
    {
        if (!initialized_) {
            ROS_WARN("Node not properly initialized, ignoring message");
            return;
        }
        
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
        } catch (...) {
            ROS_ERROR("Unknown error processing frame %d", processed_frame_count_);
        }
    }
    
    void processFrame(const sgloop_ros::SyncedFrame::ConstPtr& msg)
    {
        tic_toc_seq_.tic();
        
        // Convert ROS messages to OpenCV/Open3D format
        cv_bridge::CvImagePtr rgb_cv = cv_bridge::toCvCopy(msg->rgb, sensor_msgs::image_encodings::RGB8);
        cv_bridge::CvImagePtr depth_cv = cv_bridge::toCvCopy(msg->depth, sensor_msgs::image_encodings::TYPE_16UC1);
        cv_bridge::CvImagePtr mask_cv = cv_bridge::toCvCopy(msg->mask, sensor_msgs::image_encodings::BGR8);
        
        // Convert to Open3D images
        open3d::geometry::Image color, depth;
        color.Prepare(rgb_cv->image.cols, rgb_cv->image.rows, 3, 1);
        depth.Prepare(depth_cv->image.cols, depth_cv->image.rows, 1, 2);
        
        // Copy data
        memcpy(color.data_.data(), rgb_cv->image.data, rgb_cv->image.total() * rgb_cv->image.elemSize());
        memcpy(depth.data_.data(), depth_cv->image.data, depth_cv->image.total() * depth_cv->image.elemSize());
        
        auto rgbd = open3d::geometry::RGBDImage::CreateFromColorAndDepth(
            color, depth, global_config_->mapping_cfg.depth_scale, global_config_->mapping_cfg.depth_max, false);
        
        tic_toc_seq_.toc();
        
        // Convert pose
        Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
        pose(0, 3) = msg->pose.position.x;
        pose(1, 3) = msg->pose.position.y;
        pose(2, 3) = msg->pose.position.z;
        
        Eigen::Quaterniond q(msg->pose.orientation.w, msg->pose.orientation.x, 
                           msg->pose.orientation.y, msg->pose.orientation.z);
        pose.block<3, 3>(0, 0) = q.toRotationMatrix();
        
        // Process SAM detections (simplified for safety)
        std::vector<fmfusion::DetectionPtr> detections;
        // For now, skip complex JSON parsing to avoid crashes
        
        tic_toc_seq_.tic();
        
        // Integrate into semantic mapping
        if (semantic_mapping_) {
            semantic_mapping_->integrate(processed_frame_count_, rgbd, pose, detections);
        }
        
        tic_toc_seq_.toc();
        
        // Visualization (every 5 frames)
        if (processed_frame_count_ % 5 == 0 && semantic_mapping_) {
            try {
                Visualization::render_semantic_map(
                    semantic_mapping_->export_global_pcd(true, 0.05),
                    semantic_mapping_->export_instance_centroids(0, debug_),
                    semantic_mapping_->export_instance_annotations(0),
                    viz_,
                    LOCAL_AGENT_);
            } catch (...) {
                ROS_WARN("Visualization failed for frame %d", processed_frame_count_);
            }
        }
    }

public:
    void finalize()
    {
        if (!initialized_ || !semantic_mapping_) {
            ROS_WARN("Cannot finalize - not properly initialized");
            return;
        }
        
        ROS_WARN("Finalizing mapping with %d processed frames", processed_frame_count_);
        
        try {
            // Process at the end of the sequence
            semantic_mapping_->extract_point_cloud();
            semantic_mapping_->merge_floor(true);
            
            // Final visualization
            Visualization::render_semantic_map(
                semantic_mapping_->export_global_pcd(true, 0.05),
                semantic_mapping_->export_instance_centroids(0, debug_),
                semantic_mapping_->export_instance_annotations(0),
                viz_,
                LOCAL_AGENT_);
            
            // Save results
            if (!output_folder_.empty()) {
                ROS_WARN("Saving results to %s", output_folder_.c_str());
                semantic_mapping_->Save(output_folder_ + "/safe_online_mapping");
                tic_toc_seq_.export_data(output_folder_ + "/safe_online_mapping/time_records.txt");
                fmfusion::utility::write_config(output_folder_ + "/safe_online_mapping/config.txt", *global_config_);
            }
            
        } catch (const std::exception& e) {
            ROS_ERROR("Error during finalization: %s", e.what());
        } catch (...) {
            ROS_ERROR("Unknown error during finalization");
        }
    }
    
    bool isInitialized() const { return initialized_; }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "SafeOnlineMappingNode");
    ros::NodeHandle nh;
    ros::NodeHandle nh_private("~");
    
    SafeOnlineMappingNode node(nh, nh_private);
    
    if (!node.isInitialized()) {
        ROS_ERROR("Failed to initialize SafeOnlineMappingNode");
        return 1;
    }
    
    ros::spin();
    
    // Finalize when shutting down
    node.finalize();
    
    return 0;
}
