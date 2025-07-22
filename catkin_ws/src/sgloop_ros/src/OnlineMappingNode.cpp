#include <iostream>
#include <memory>
#include <vector>
#include <fstream>

#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include "tf/transform_listener.h"

// ROS messages
#include <sensor_msgs/Image.h>
#include <geometry_msgs/Pose.h>
#include <std_msgs/String.h>
#include <std_msgs/Header.h>
#include "sgloop_ros/SyncedFrame.h"

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

    // Simple subscriber for synchronized data
    ros::Subscriber synced_frame_sub_;
    
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
          tic_toc_seq_("# Online Mapping", 3)
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
        
        // Initialize subscriber for synchronized data
        synced_frame_sub_ = nh_.subscribe("/synced_frame", 1, &OnlineMappingNode::syncedFrameCallback, this);
        
        ROS_INFO("OnlineMappingNode initialized. Waiting for synchronized messages...");
    }
    
    ~OnlineMappingNode()
    {
        if (semantic_mapping_) delete semantic_mapping_;
        if (global_config_) delete global_config_;
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
        tic_toc_seq_.tic();

        try {
            processFrame(msg);
            processed_frame_count_++;
        } catch (const std::exception& e) {
            ROS_ERROR("Error processing frame %d: %s", processed_frame_count_, e.what());
        }
    }

    void processFrame(const sgloop_ros::SyncedFrame::ConstPtr& msg)
    {
        // Convert ROS messages to OpenCV/Open3D format
        cv_bridge::CvImagePtr rgb_cv = cv_bridge::toCvCopy(msg->rgb, sensor_msgs::image_encodings::RGB8);
        cv_bridge::CvImagePtr depth_cv = cv_bridge::toCvCopy(msg->depth, sensor_msgs::image_encodings::TYPE_16UC1);
        cv_bridge::CvImagePtr mask_cv = cv_bridge::toCvCopy(msg->mask, sensor_msgs::image_encodings::MONO8);

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

        // Convert pose (format is already consistent with offline version)
        Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
        pose(0, 3) = msg->pose.position.x;
        pose(1, 3) = msg->pose.position.y;
        pose(2, 3) = msg->pose.position.z;

        Eigen::Quaterniond q(msg->pose.orientation.w, msg->pose.orientation.x,
                           msg->pose.orientation.y, msg->pose.orientation.z);
        pose.block<3, 3>(0, 0) = q.toRotationMatrix();

        // Process SAM detections from JSON
        std::vector<fmfusion::DetectionPtr> detections;
        bool detections_loaded = parseJSONDetections(mask_cv->image, msg->json, detections);

        if (!detections_loaded) {
            ROS_WARN("Failed to process SAM detections for frame %d", processed_frame_count_);
            // Continue with empty detections
        } else {
            ROS_INFO("Processed %zu SAM detections for frame %d", detections.size(), processed_frame_count_);
        }

        tic_toc_seq_.tic();

        // Integrate into semantic mapping (same as offline version)
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
    }

    bool parseJSONDetections(const cv::Mat& mask_image,
                           const std::string& json_data,
                           std::vector<fmfusion::DetectionPtr>& detections)
    {
        try {
            detections.clear();

            if (mask_image.empty() || json_data.empty()) {
                ROS_WARN("Empty mask image or JSON data");
                return false;
            }

            // Simple JSON parsing without external library
            // Parse the JSON array format: [{"value": 1, "label": "person", "box": [...], "logit": 0.7}, ...]

            // For now, create detections from mask values and use simple string parsing for labels
            // This avoids jsoncpp dependency issues

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
                auto detection = std::make_shared<fmfusion::Detection>(mask_value);

                // Try to extract label from JSON for this mask value
                std::string label = extractLabelFromJSON(json_data, mask_value);
                float confidence = extractConfidenceFromJSON(json_data, mask_value);

                detection->labels_.push_back(std::make_pair(label, confidence));

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

                detection->instances_idxs_ = instance_mask.clone();
                detections.push_back(detection);
            }

            return true;

        } catch (const std::exception& e) {
            ROS_ERROR("Error parsing JSON detections: %s", e.what());
            return false;
        }
    }

    std::string extractLabelFromJSON(const std::string& json_data, int mask_value)
    {
        // Simple string search for the label corresponding to mask_value
        std::string search_pattern = "\"value\": " + std::to_string(mask_value);
        size_t pos = json_data.find(search_pattern);

        if (pos != std::string::npos) {
            // Look for the label field after the value
            size_t label_pos = json_data.find("\"label\":", pos);
            if (label_pos != std::string::npos) {
                size_t start = json_data.find("\"", label_pos + 8) + 1;
                size_t end = json_data.find("\"", start);
                if (start != std::string::npos && end != std::string::npos) {
                    return json_data.substr(start, end - start);
                }
            }
        }

        return "object_" + std::to_string(mask_value);
    }

    float extractConfidenceFromJSON(const std::string& json_data, int mask_value)
    {
        // Simple string search for the logit corresponding to mask_value
        std::string search_pattern = "\"value\": " + std::to_string(mask_value);
        size_t pos = json_data.find(search_pattern);

        if (pos != std::string::npos) {
            // Look for the logit field after the value
            size_t logit_pos = json_data.find("\"logit\":", pos);
            if (logit_pos != std::string::npos) {
                size_t start = logit_pos + 8;
                size_t end = json_data.find_first_of(",}", start);
                if (end != std::string::npos) {
                    std::string logit_str = json_data.substr(start, end - start);
                    try {
                        return std::stof(logit_str);
                    } catch (...) {
                        // Ignore parsing errors
                    }
                }
            }
        }

        return 0.5f; // Default confidence
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
