#include <iostream>
#include <memory>
#include <vector>
#include <fstream>

#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include "tf/transform_listener.h"

// Message filters for time synchronization
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

// ROS messages
#include <sensor_msgs/Image.h>
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/String.h>
#include <std_msgs/Header.h>

#include <open3d/Open3D.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sstream>

#include "tools/Utility.h"
#include "tools/IO.h"
#include "tools/TicToc.h"
#include "mapping/SemanticMapping.h"

#include "Visualization.h"

class OnlineMappingNode
{
private:
    ros::NodeHandle nh_;
    ros::NodeHandle nh_private_;
    
    // Message filters for time synchronization (only for images)
    message_filters::Subscriber<sensor_msgs::Image> rgb_sub_;
    message_filters::Subscriber<sensor_msgs::Image> depth_sub_;
    message_filters::Subscriber<sensor_msgs::Image> mask_sub_;

    typedef message_filters::sync_policies::ApproximateTime<
        sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image> SyncPolicy;
    typedef message_filters::Synchronizer<SyncPolicy> Synchronizer;

    std::shared_ptr<Synchronizer> sync_;

    // Separate subscribers for pose and JSON
    ros::Subscriber pose_sub_;
    ros::Subscriber json_sub_;

    // Latest pose and JSON data
    geometry_msgs::PoseStamped latest_pose_;
    std_msgs::String latest_json_;
    bool pose_received_;
    bool json_received_;
    
    // FM-Fusion components
    fmfusion::Config *global_config_;
    fmfusion::SemanticMapping *semantic_mapping_;
    Visualization::Visualizer viz_;
    
    // Parameters
    std::string LOCAL_AGENT_;
    std::string output_folder_;
    int frame_gap_;
    int max_frames_;
    bool debug_;
    int frame_count_;
    int processed_frame_count_;
    
    fmfusion::TicTocSequence tic_toc_seq_;

public:
    OnlineMappingNode(ros::NodeHandle& nh, ros::NodeHandle& nh_private)
        : nh_(nh), nh_private_(nh_private), viz_(nh, nh_private),
          frame_count_(0), processed_frame_count_(0),
          tic_toc_seq_("# Online Mapping", 3), pose_received_(false), json_received_(false)
    {
        // Load parameters
        std::string config_file;
        assert(nh_private_.getParam("cfg_file", config_file));
        nh_private_.getParam("local_agent", LOCAL_AGENT_);
        frame_gap_ = nh_private_.param("frame_gap", 1);
        output_folder_ = nh_private_.param("output_folder", std::string(""));
        int o3d_verbose_level = nh_private_.param("o3d_verbose_level", 2);
        int visualization = nh_private_.param("visualization", 0);
        max_frames_ = nh_private_.param("max_frames", 5000);
        debug_ = nh_private_.param("debug", false);
        
        ROS_WARN("OnlineMappingNode started");
        
        // Initialize FM-Fusion
        global_config_ = fmfusion::utility::create_scene_graph_config(config_file, true);
        open3d::utility::SetVerbosityLevel((open3d::utility::VerbosityLevel)o3d_verbose_level);
        
        if(output_folder_.size() > 0 && !open3d::utility::filesystem::DirectoryExists(output_folder_)) {
            open3d::utility::filesystem::MakeDirectory(output_folder_);
        }
        
        std::ofstream out_file(output_folder_ + "/config.txt");
        out_file << fmfusion::utility::config_to_message(*global_config_);
        out_file.close();
        
        semantic_mapping_ = new fmfusion::SemanticMapping(global_config_->mapping_cfg, global_config_->instance_cfg);
        
        // Initialize subscribers
        rgb_sub_.subscribe(nh_, "/camera/color/image_raw", 1);
        depth_sub_.subscribe(nh_, "/camera/depth/image_rect_raw", 1);
        mask_sub_.subscribe(nh_, "/mask_image", 1);

        // Separate subscribers for pose and JSON
        pose_sub_ = nh_.subscribe("/vins_estimator/camera_pose", 10, &OnlineMappingNode::poseCallback, this);
        json_sub_ = nh_.subscribe("/mask_data", 10, &OnlineMappingNode::jsonCallback, this);

        // Initialize synchronizer (only for images)
        sync_.reset(new Synchronizer(SyncPolicy(10), rgb_sub_, depth_sub_, mask_sub_));
        sync_->registerCallback(boost::bind(&OnlineMappingNode::syncCallback, this, _1, _2, _3));
        
        ROS_INFO("OnlineMappingNode initialized. Waiting for synchronized messages...");
    }
    
    ~OnlineMappingNode()
    {
        if (semantic_mapping_) delete semantic_mapping_;
        if (global_config_) delete global_config_;
    }

private:
    void poseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg)
    {
        latest_pose_ = *msg;
        pose_received_ = true;
    }

    void jsonCallback(const std_msgs::String::ConstPtr& msg)
    {
        latest_json_ = *msg;
        json_received_ = true;
    }

    bool processSAMDetections(const cv::Mat& mask_image,
                             const std::string& json_data,
                             std::vector<fmfusion::DetectionPtr>& detections)
    {
        try {
            // Clear previous detections
            detections.clear();

            if (mask_image.empty()) {
                ROS_WARN("Empty mask image");
                return false;
            }

            // For now, create simple detections from mask without JSON parsing
            // This avoids the jsoncpp version conflict issue
            // TODO: Fix JSON parsing when jsoncpp version is resolved

            // Find unique mask values (excluding 0 which is background)
            std::set<int> unique_values;
            for (int y = 0; y < mask_image.rows; ++y) {
                for (int x = 0; x < mask_image.cols; ++x) {
                    int val = mask_image.at<uchar>(y, x);
                    if (val > 0) {
                        unique_values.insert(val);
                    }
                }
            }

            // Create detection for each unique mask value
            for (int mask_value : unique_values) {
                // Create detection with proper constructor
                auto detection = std::make_shared<fmfusion::Detection>(mask_value);

                // Set default label
                std::string label = "object_" + std::to_string(mask_value);
                detection->labels_.push_back(std::make_pair(label, 0.8f));

                // Calculate bounding box from mask
                cv::Mat instance_mask = (mask_image == mask_value);
                std::vector<cv::Point> points;
                cv::findNonZero(instance_mask, points);
                if (!points.empty()) {
                    cv::Rect bbox = cv::boundingRect(points);
                    detection->bbox_.u0 = bbox.x;
                    detection->bbox_.v0 = bbox.y;
                    detection->bbox_.u1 = bbox.x + bbox.width;
                    detection->bbox_.v1 = bbox.y + bbox.height;
                }

                // Extract mask for this ID
                detection->instances_idxs_ = instance_mask.clone();

                detections.push_back(detection);
            }

            ROS_INFO("Created %zu detections from mask", detections.size());
            return true;

        } catch (const std::exception& e) {
            ROS_ERROR("Error processing SAM detections: %s", e.what());
            return false;
        }
    }

    void syncCallback(const sensor_msgs::ImageConstPtr& rgb_msg,
                     const sensor_msgs::ImageConstPtr& depth_msg,
                     const sensor_msgs::ImageConstPtr& mask_msg)
    {
        frame_count_++;

        // Check if we have pose and JSON data
        if (!pose_received_ || !json_received_) {
            ROS_WARN("Waiting for pose and JSON data...");
            return;
        }

        // Apply frame gap
        if ((frame_count_ - 1) % frame_gap_ != 0) {
            return;
        }

        if (processed_frame_count_ >= max_frames_) {
            ROS_WARN("Reached maximum frames (%d), stopping processing", max_frames_);
            return;
        }

        ROS_INFO("Processing frame %d (total received: %d)...", processed_frame_count_, frame_count_);
        tic_toc_seq_.tic();
        
        try {
            // Convert ROS messages to OpenCV/Open3D format
            cv_bridge::CvImagePtr rgb_cv = cv_bridge::toCvCopy(rgb_msg, sensor_msgs::image_encodings::RGB8);
            cv_bridge::CvImagePtr depth_cv = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_16UC1);
            cv_bridge::CvImagePtr mask_cv = cv_bridge::toCvCopy(mask_msg, sensor_msgs::image_encodings::MONO8);
            
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
            pose(0, 3) = latest_pose_.pose.position.x;
            pose(1, 3) = latest_pose_.pose.position.y;
            pose(2, 3) = latest_pose_.pose.position.z;

            Eigen::Quaterniond q(latest_pose_.pose.orientation.w, latest_pose_.pose.orientation.x,
                               latest_pose_.pose.orientation.y, latest_pose_.pose.orientation.z);
            pose.block<3, 3>(0, 0) = q.toRotationMatrix();

            // Process SAM detections
            std::vector<fmfusion::DetectionPtr> detections;
            bool detections_loaded = processSAMDetections(mask_cv->image, latest_json_.data, detections);

            if (!detections_loaded) {
                ROS_WARN("Failed to process SAM detections for frame %d", processed_frame_count_);
                // Continue with empty detections
            } else {
                ROS_INFO("Processed %zu SAM detections for frame %d", detections.size(), processed_frame_count_);
            }
            
            tic_toc_seq_.tic();
            
            // Integrate into semantic mapping
            semantic_mapping_->integrate(processed_frame_count_, rgbd, pose, detections);
            
            tic_toc_seq_.toc();
            
            // Visualization
            if (processed_frame_count_ % 10 == 0) {
                Visualization::render_semantic_map(
                    semantic_mapping_->export_global_pcd(true, 0.05),
                    semantic_mapping_->export_instance_centroids(0, debug_),
                    semantic_mapping_->export_instance_annotations(0),
                    viz_,
                    LOCAL_AGENT_);
            }
            
            processed_frame_count_++;
            
        } catch (const std::exception& e) {
            ROS_ERROR("Error processing frame %d: %s", processed_frame_count_, e.what());
        }
    }

public:
    void finalize()
    {
        ROS_WARN("Finalizing mapping with %d processed frames", processed_frame_count_);
        
        // Process at the end of the sequence
        semantic_mapping_->extract_point_cloud();
        semantic_mapping_->merge_floor(true);
        
        Visualization::render_semantic_map(
            semantic_mapping_->export_global_pcd(true, 0.05),
            semantic_mapping_->export_instance_centroids(0, debug_),
            semantic_mapping_->export_instance_annotations(0),
            viz_,
            LOCAL_AGENT_);
        
        // Save results
        if (!output_folder_.empty()) {
            ROS_WARN("Saving results to %s", output_folder_.c_str());
            semantic_mapping_->Save(output_folder_ + "/online_mapping");
            tic_toc_seq_.export_data(output_folder_ + "/online_mapping/time_records.txt");
            fmfusion::utility::write_config(output_folder_ + "/online_mapping/config.txt", *global_config_);
        }
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "OnlineMappingNode");
    ros::NodeHandle nh;
    ros::NodeHandle nh_private("~");
    
    OnlineMappingNode node(nh, nh_private);
    
    ros::spin();
    
    // Finalize when shutting down
    node.finalize();
    
    return 0;
}
