#pragma once

#include <string>
#include "DataType.h"
#include <opencv2/core.hpp>      // cv::Mat
#include <opencv2/features2d.hpp> // cv::KeyPoint, cv::DMatch
#include <vector>

struct Status {
 bool success = true;
 std::string message = "OK";
};

struct Config
{
    Config(){};
    Config(const std::string& yaml_path);
    std::string input_img_dir_;
    std::string camera_extrinsic_path_;
    std::string camera_intrinsic_path_;
    std::string camera_timestamp_path_;
    std::string projective_z_buffer_dir_;
    // don't include LIO_results 
    std::string lidar_ply_dirs_;
    std::string lidar_kf_path_;
    std::string output_path_;

    // SIFT parameters.
    int sift_nfeatures_ = 0;
    int sift_n_octave_layers_ = 3;
    double sift_contrast_threshold_ = 0.04;
    double sift_edge_threshold_ = 10.0;
    double sift_sigma_ = 1.6;

    // Matching.
    double ratio_threshold_ = 0.75;
    double ransac_threshold_ = 1.0;
    int min_inliers_ = 8;

    int num_iteration_ =20;

    // Bundle adjustment convergence.
    // Max iterations for each Ceres solve.
    int ceres_max_iterations_ = 50;
    // Outer refinement loop stops early once the relative change in final
    // cost between iterations drops below this threshold.
    double ba_cost_threshold_ = 1e-4;
};

struct LidarPointCloudInfo {
    int lidar_id_;
    std::string lidar_path_;
    double time_stamp_;
    // lidar --> world
    SE3d initial_T_wl_;
};

struct Observation {
    int camera_id_;
    int keypoint_idx_;

    Vec2d pixel_;
    double depth_ = 0.0;
    double optimized_depth_ = 0.0;
    bool optimized_ = false;
};

struct Landmark {
    int landmark_id_;
    bool optimized_ =false;

    Vec3d initial_position_;
    Vec3d optimized_position_;


    std::vector<Observation> observations_;
};

struct CameraIntrinsic {
    Mat3d K = Eigen::Matrix3d::Identity();

    double fx() const {
        return K(0, 0);
    }

    double fy() const {
        return K(1, 1);
    }

    double cx() const {
        return K(0, 2);
    }

    double cy() const {
        return K(1, 2);
    }
};

struct CameraExtrinsic {
    // Transform a LiDAR point into the camera coordinate system:
    //
    // p_C = T_camera_lidar * p_L
    SE3d T_camera_lidar;

    // Transform a camera point into the LiDAR coordinate system:
    //
    // p_L = T_lidar_camera * p_C
    SE3d T_lidar_camera;
};
struct Camera{
 int camera_id_;
 std::string camera_name_;
 std::string camera_path_;
 double time_stamp_;
 SE3d initial_T_cw_;
 SE3d optimized_T_cw_;
 bool optimized_ = false;
 int matched_lidar_id_ = -1;
 int img_width_;
 int img_height_;
 // add depth or descriptor
 cv::Mat depth_map_;
 cv::Mat descriptors_;
 std::vector<cv::KeyPoint> keypoints_;
};