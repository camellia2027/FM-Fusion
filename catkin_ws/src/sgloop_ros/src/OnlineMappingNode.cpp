#include <iostream>
#include <memory>
#include <sys/stat.h>
#include <vector>
#include <fstream>
#include <signal.h>

#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include "tf/transform_listener.h"
#include <sensor_msgs/Image.h>
#include <geometry_msgs/Pose.h>
#include <cv_bridge/cv_bridge.h>

#include <open3d/Open3D.h>
#include <opencv2/opencv.hpp>
#include <json/json.h>

#include "tools/Utility.h"
#include "tools/IO.h"
#include "tools/TicToc.h"
#include "mapping/SemanticMapping.h"
#include "mapping/Detection.h"

#include "Visualization.h"
#include "sgloop_ros/SyncedFrame.h"

class OnlineMappingNode
{
private:
    ros::NodeHandle nh_;
    ros::NodeHandle nh_private_;
    ros::Subscriber synced_frame_sub_;
    
    // FM-Fusion components
    fmfusion::Config* global_config_;
    fmfusion::SemanticMapping* semantic_mapping_;
    Visualization::Visualizer viz_;
    
    // Parameters
    std::string config_file_;
    std::string output_folder_;
    std::string LOCAL_AGENT_;
    int frame_gap_;
    int max_frames_;
    bool debug_;
    
    // State
    int frame_count_;
    int prev_frame_id_;
    fmfusion::TicTocSequence tic_toc_seq_;
    
public:
    OnlineMappingNode(ros::NodeHandle& nh, ros::NodeHandle& nh_private) 
        : nh_(nh), nh_private_(nh_private), viz_(nh, nh_private), 
          frame_count_(0), prev_frame_id_(-100),
          tic_toc_seq_("# Online Mapping", 3)
    {
        // 获取参数
        if (!nh_private_.getParam("cfg_file", config_file_)) {
            config_file_ = "config/your_camera.yaml";
        }
        
        nh_private_.getParam("local_agent", LOCAL_AGENT_);
        frame_gap_ = nh_private_.param("frame_gap", 1);
        output_folder_ = nh_private_.param("output_folder", std::string(""));
        max_frames_ = nh_private_.param("max_frames", 5000);
        debug_ = nh_private_.param("debug", false);
        
        ROS_INFO("OnlineMappingNode started with config: %s", config_file_.c_str());
        
        // 使用修复后的配置文件进行FM-Fusion初始化
        ROS_INFO("Starting FM-Fusion initialization with fixed config...");
        initializeFMFusionGradual();
        
        // 订阅同步帧数据
        synced_frame_sub_ = nh_.subscribe("/sync/output", 10,
                                         &OnlineMappingNode::syncedFrameCallback, this);
        
        ROS_INFO("OnlineMappingNode initialized successfully");
    }
    
    ~OnlineMappingNode()
    {
        if (semantic_mapping_) {
            delete semantic_mapping_;
        }
        if (global_config_) {
            delete global_config_;
        }
    }
    
private:
    void initializeFMFusion()
    {
        ROS_INFO("Starting FM-Fusion initialization...");

        // 加载配置 - 使用与MappingNode完全相同的方式
        ROS_INFO("Loading config file: %s", config_file_.c_str());
        try {
            global_config_ = fmfusion::utility::create_scene_graph_config(config_file_, true);
            if (!global_config_) {
                ROS_ERROR("Failed to load config file: %s", config_file_.c_str());
                return;
            }
            ROS_INFO("Config loaded successfully");

            // 设置Open3D详细级别 - 与MappingNode相同
            open3d::utility::SetVerbosityLevel((open3d::utility::VerbosityLevel)2);
            ROS_INFO("Open3D verbosity level set");

            // 设置输出目录 - 与MappingNode相同的逻辑
            if (output_folder_.size() > 0 && !open3d::utility::filesystem::DirectoryExists(output_folder_)) {
                open3d::utility::filesystem::MakeDirectory(output_folder_);
                ROS_INFO("Created output directory: %s", output_folder_.c_str());
            }

            // 写入配置文件 - 与MappingNode相同
            if (!output_folder_.empty()) {
                std::ofstream out_file(output_folder_ + "/config.txt");
                out_file << fmfusion::utility::config_to_message(*global_config_);
                out_file.close();
                ROS_INFO("Config file written to output directory");
            }

            // 初始化语义建图 - 使用与MappingNode完全相同的方式
            ROS_INFO("Initializing SemanticMapping...");
            semantic_mapping_ = new fmfusion::SemanticMapping(global_config_->mapping_cfg,
                                                              global_config_->instance_cfg);
            ROS_INFO("SemanticMapping initialized");

            ROS_INFO("FM-Fusion initialized successfully");
        } catch (const std::exception& e) {
            ROS_ERROR("Exception during FM-Fusion initialization: %s", e.what());
            throw;
        }
    }
    
    void syncedFrameCallback(const sgloop_ros::SyncedFrame::ConstPtr& msg)
    {
        if (frame_count_ >= max_frames_) {
            return;
        }

        if ((frame_count_ - prev_frame_id_) < frame_gap_) {
            frame_count_++;
            return;
        }

        ROS_INFO("Processing frame %d...", frame_count_);
        tic_toc_seq_.tic();

        // 转换ROS图像消息到Open3D格式
        open3d::geometry::Image color, depth;

        // 转换RGB图像
        cv_bridge::CvImagePtr cv_rgb;
        try {
            cv_rgb = cv_bridge::toCvCopy(msg->rgb, sensor_msgs::image_encodings::BGR8);
            // 转换BGR到RGB
            cv::Mat rgb_mat;
            cv::cvtColor(cv_rgb->image, rgb_mat, cv::COLOR_BGR2RGB);

            // 转换为Open3D格式
            color.Prepare(rgb_mat.cols, rgb_mat.rows, 3, 1);
            memcpy(color.data_.data(), rgb_mat.data, rgb_mat.total() * rgb_mat.elemSize());
        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception for RGB: %s", e.what());
            return;
        }

        // 转换深度图像
        cv_bridge::CvImagePtr cv_depth;
        try {
            cv_depth = cv_bridge::toCvCopy(msg->depth, sensor_msgs::image_encodings::TYPE_16UC1);

            // 转换为Open3D格式
            depth.Prepare(cv_depth->image.cols, cv_depth->image.rows, 1, 2);
            memcpy(depth.data_.data(), cv_depth->image.data, cv_depth->image.total() * cv_depth->image.elemSize());
        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception for depth: %s", e.what());
            return;
        }

        // 创建RGBD图像
        auto rgbd = open3d::geometry::RGBDImage::CreateFromColorAndDepth(
            color, depth,
            global_config_->mapping_cfg.depth_scale,
            global_config_->mapping_cfg.depth_max,
            false);

        tic_toc_seq_.toc();

        // 解析检测数据
        ROS_INFO("=== Frame %d JSON Data Debug ===", frame_count_);
        ROS_INFO("JSON length: %zu characters", msg->json.length());
        ROS_INFO("JSON first 200 chars: %s", msg->json.substr(0, 200).c_str());
        ROS_INFO("Mask image size: %dx%d, encoding: %s",
                 msg->mask.width, msg->mask.height, msg->mask.encoding.c_str());

        std::vector<fmfusion::DetectionPtr> detections;
        bool loaded = parseDetectionsFromJson(msg->json, msg->mask, detections);

        ROS_INFO("JSON parsing result: %s, detections count: %zu",
                 loaded ? "SUCCESS" : "FAILED", detections.size());

        if (!loaded) {
            ROS_WARN("Failed to parse detections from JSON for frame %d", frame_count_);
            frame_count_++;
            return;
        }

        // 转换位姿
        Eigen::Matrix4d pose = transformMatrixToPose(msg->transform_matrix);

        // 进行语义建图
        semantic_mapping_->integrate(frame_count_, rgbd, pose, detections);
        tic_toc_seq_.toc();

        // 可视化
        visualizeResults(pose, color, detections);

        prev_frame_id_ = frame_count_;
        frame_count_++;

        tic_toc_seq_.toc();

        ROS_INFO("Frame %d processed successfully", frame_count_ - 1);
    }

    void initializeFMFusionGradual()
    {
        ROS_INFO("Step 0: Starting initialization");
        ROS_INFO("Step 0.1: config_file_ length: %zu", config_file_.length());

        if (config_file_.empty()) {
            ROS_ERROR("Config file path is empty!");
            return;
        }

        ROS_INFO("Step 1: Loading config file: %s", config_file_.c_str());

        // 添加文件存在性检查
        ROS_INFO("Step 1.0: Checking file existence...");
        std::ifstream test_file(config_file_);
        if (!test_file.good()) {
            ROS_ERROR("Config file does not exist or cannot be read: %s", config_file_.c_str());
            return;
        }
        test_file.close();
        ROS_INFO("Step 1.0: File check passed");
        
        try {
            ROS_INFO("Step 1.1: Calling create_scene_graph_config...");
            global_config_ = fmfusion::utility::create_scene_graph_config(config_file_, true);
            ROS_INFO("Step 1.2: create_scene_graph_config returned successfully");
            
            if (!global_config_) {
                ROS_ERROR("Failed to load config file: %s", config_file_.c_str());
                return;
            }
            ROS_INFO("Step 1: Config loaded successfully");

            ROS_INFO("Step 2: Setting Open3D verbosity level");
            open3d::utility::SetVerbosityLevel((open3d::utility::VerbosityLevel)2);
            ROS_INFO("Step 2: Open3D verbosity level set");

            ROS_INFO("Step 3: Creating output directory");
            if (output_folder_.size() > 0 && !open3d::utility::filesystem::DirectoryExists(output_folder_)) {
                open3d::utility::filesystem::MakeDirectory(output_folder_);
                ROS_INFO("Step 3: Created output directory: %s", output_folder_.c_str());
            }

            ROS_INFO("Step 4: Writing config file");
            if (!output_folder_.empty()) {
                std::ofstream out_file(output_folder_ + "/config.txt");
                out_file << fmfusion::utility::config_to_message(*global_config_);
                out_file.close();
                ROS_INFO("Step 4: Config file written to output directory");
            }

            ROS_INFO("Step 5: Initializing SemanticMapping");
            semantic_mapping_ = new fmfusion::SemanticMapping(global_config_->mapping_cfg,
                                                              global_config_->instance_cfg);
            ROS_INFO("Step 5: SemanticMapping initialized successfully");

            ROS_INFO("FM-Fusion gradual initialization completed successfully");
        } catch (const std::exception& e) {
            ROS_ERROR("Exception during gradual FM-Fusion initialization: %s", e.what());
            throw;
        }
    }

    bool parseDetectionsFromJson(const std::string& json_str,
                                const sensor_msgs::Image& mask_msg,
                                std::vector<fmfusion::DetectionPtr>& detections)
    {
        detections.clear();

        ROS_INFO("=== parseDetectionsFromJson Debug ===");
        ROS_INFO("JSON string length: %zu", json_str.length());
        ROS_INFO("JSON string (first 500 chars): %s", json_str.substr(0, 500).c_str());

        if (json_str.empty()) {
            ROS_WARN("JSON string is empty, returning true with no detections");
            return true; // 允许空检测
        }

        try {
            Json::Value root;
            Json::Reader reader;
            if (!reader.parse(json_str, root)) {
                ROS_ERROR("Failed to parse JSON: %s", reader.getFormattedErrorMessages().c_str());
                return false;
            }

            ROS_INFO("JSON parsed successfully, root type: %d", root.type());

            // 检查JSON结构
            ROS_INFO("JSON root members:");
            for (const auto& member : root.getMemberNames()) {
                ROS_INFO("  - %s (type: %d)", member.c_str(), root[member].type());
            }

            // 解析newdata.bag中的JSON格式
            if (!root.isMember("detections") || !root["detections"].isArray()) {
                ROS_ERROR("JSON does not contain valid 'detections' array");
                ROS_ERROR("Available members: %s", Json::writeString(Json::StreamWriterBuilder(), root).c_str());
                return false;
            }

            const Json::Value& detections_array = root["detections"];
            ROS_INFO("Found detections array with %d elements", detections_array.size());

            if (detections_array.size() < 1) {
                ROS_WARN("No detections found in JSON.");
                return true; // 返回true但检测为空
            }

            // 获取掩码图像
            cv_bridge::CvImagePtr cv_mask;
            try {
                // 先尝试直接转换，如果失败则转换编码
                if (mask_msg.encoding == sensor_msgs::image_encodings::MONO8) {
                    cv_mask = cv_bridge::toCvCopy(mask_msg, sensor_msgs::image_encodings::MONO8);
                } else {
                    // 如果是BGR8或其他格式，提取第一个通道作为ID图像
                    cv_bridge::CvImagePtr cv_temp = cv_bridge::toCvCopy(mask_msg);
                    cv_mask = boost::make_shared<cv_bridge::CvImage>();
                    cv_mask->header = cv_temp->header;
                    cv_mask->encoding = sensor_msgs::image_encodings::MONO8;

                    if (cv_temp->image.channels() == 3) {
                        // 对于BGR图像，使用第一个通道（B通道）作为ID图像
                        // 注意：OpenCV中BGR顺序是[B,G,R]，第一个通道是B通道
                        std::vector<cv::Mat> channels;
                        cv::split(cv_temp->image, channels);
                        cv_mask->image = channels[0].clone(); // 使用B通道
                        ROS_INFO("Using first channel (B) of BGR mask as ID image");
                    } else if (cv_temp->image.channels() == 1) {
                        cv_mask->image = cv_temp->image.clone();
                    } else {
                        ROS_ERROR("Unsupported mask image channels: %d", cv_temp->image.channels());
                        return false;
                    }
                }
                ROS_INFO("Mask image converted successfully: %dx%d, unique values: %d",
                         cv_mask->image.cols, cv_mask->image.rows,
                         cv::countNonZero(cv_mask->image != cv_mask->image.at<uchar>(0,0)));
            } catch (cv_bridge::Exception& e) {
                ROS_ERROR("cv_bridge exception for mask: %s", e.what());
                return false;
            }

            // 解析每个检测对象 - 基于newdata.bag的实际格式
            for (int i = 0; i < detections_array.size(); ++i) {
                const Json::Value& detection_json = detections_array[i];
                int detection_id = detection_json["value"].asInt();

                if (detection_id == 0) continue; // 跳过背景

                auto detection = std::make_shared<fmfusion::Detection>(detection_id);

                // 解析标签和分数 - newdata.bag格式
                if (detection_json.isMember("label")) {
                    std::string label = detection_json["label"].asString();
                    float score = 1.0f; // 默认分数

                    if (detection_json.isMember("logit")) {
                        score = detection_json["logit"].asFloat();
                    }

                    detection->labels_.push_back(std::make_pair(label, score));
                }

                // 解析边界框 - newdata.bag格式
                if (detection_json.isMember("box") && detection_json["box"].isArray() &&
                    detection_json["box"].size() == 4) {
                    const Json::Value& box = detection_json["box"];
                    detection->bbox_.u0 = box[0].asDouble();
                    detection->bbox_.v0 = box[1].asDouble();
                    detection->bbox_.u1 = box[2].asDouble();
                    detection->bbox_.v1 = box[3].asDouble();
                } else {
                    // 如果没有边界框信息，设置默认值
                    detection->bbox_.u0 = 0;
                    detection->bbox_.v0 = 0;
                    detection->bbox_.u1 = cv_mask->image.cols;
                    detection->bbox_.v1 = cv_mask->image.rows;
                }

                // 从掩码图像中提取对应的实例掩码（如果有掩码图像）
                if (cv_mask && !cv_mask->image.empty()) {
                    cv::Mat instance_mask = (cv_mask->image == detection_id);
                    detection->instances_idxs_ = instance_mask.clone();

                    // 如果JSON中没有边界框信息，从掩码中计算
                    if (!detection_json.isMember("box")) {
                        std::vector<std::vector<cv::Point>> contours;
                        cv::findContours(instance_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

                        if (!contours.empty()) {
                            cv::Rect bbox = cv::boundingRect(contours[0]);
                            for (size_t j = 1; j < contours.size(); ++j) {
                                bbox |= cv::boundingRect(contours[j]);
                            }

                            detection->bbox_.u0 = bbox.x;
                            detection->bbox_.v0 = bbox.y;
                            detection->bbox_.u1 = bbox.x + bbox.width;
                            detection->bbox_.v1 = bbox.y + bbox.height;
                        }
                    }
                } else {
                    // 如果没有掩码图像，创建基于边界框的简单掩码
                    if (detection_json.isMember("box")) {
                        cv::Mat simple_mask = cv::Mat::zeros(480, 640, CV_8UC1); // 假设图像尺寸
                        cv::Rect roi(detection->bbox_.u0, detection->bbox_.v0,
                                   detection->bbox_.u1 - detection->bbox_.u0,
                                   detection->bbox_.v1 - detection->bbox_.v0);
                        simple_mask(roi) = 255;
                        detection->instances_idxs_ = simple_mask;
                    }
                }

                detections.push_back(detection);
            }

            ROS_INFO("Parsed %zu detections from JSON", detections.size());
            return true;

        } catch (const std::exception& e) {
            ROS_ERROR("Exception parsing JSON: %s", e.what());
            return false;
        }
    }
    
    template<typename T>
    Eigen::Matrix4d transformMatrixToPose(const T& transform_array)
    {
        Eigen::Matrix4d matrix = Eigen::Matrix4d::Identity();

        if (transform_array.size() == 16) {
            // 将数组转换为4x4矩阵
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    matrix(i, j) = transform_array[i * 4 + j];
                }
            }
        } else {
            ROS_WARN("Invalid transform matrix size: %zu, expected 16", transform_array.size());
        }

        return matrix;
    }

    
    void visualizeResults(const Eigen::Matrix4d& pose, 
                         const open3d::geometry::Image& color,
                         const std::vector<fmfusion::DetectionPtr>& detections)
    {
        // 可视化相机位姿
        Visualization::render_camera_pose(pose, viz_.camera_pose, LOCAL_AGENT_, frame_count_);
        Visualization::render_path(pose, viz_.path_msg, viz_.path, LOCAL_AGENT_, frame_count_);
        
        // 可视化检测结果
        if (viz_.pred_image.getNumSubscribers() > 0) {
            Visualization::render_rgb_detections(color, detections, viz_.pred_image, LOCAL_AGENT_);
        }
        
        // 可视化3D语义地图
        Visualization::render_semantic_map(
            semantic_mapping_->export_global_pcd(true, 0.05),
            semantic_mapping_->export_instance_centroids(0, debug_),
            semantic_mapping_->export_instance_annotations(0),
            viz_,
            LOCAL_AGENT_);
    }
    
public:
    void saveResults()
    {
        if (output_folder_.empty()) {
            return;
        }
        
        ROS_INFO("Saving results to %s", output_folder_.c_str());
        
        // 最终处理
        semantic_mapping_->extract_point_cloud();
        semantic_mapping_->merge_floor(true);
        
        // 保存结果
        std::string sequence_name = "online_mapping";
        semantic_mapping_->Save(output_folder_ + "/" + sequence_name);
        tic_toc_seq_.export_data(output_folder_ + "/" + sequence_name + "/time_records.txt");
        fmfusion::utility::write_config(output_folder_ + "/" + sequence_name + "/config.txt", *global_config_);
        
        ROS_INFO("Results saved successfully");
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "online_mapping_node");
    ros::NodeHandle nh;
    ros::NodeHandle nh_private("~");
    
    OnlineMappingNode mapping_node(nh, nh_private);
    
    ROS_INFO("Online mapping node started. Waiting for synced frames...");
    
    // 设置关闭时保存结果的信号处理
    // ros::on_shutdown在某些版本中可能不可用，使用signal处理
    signal(SIGINT, [](int sig) {
        ROS_INFO("Received SIGINT, shutting down...");
        ros::shutdown();
    });
    
    ros::spin();
    
    return 0;
}
