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
{   Config(const std::string& yaml_path);
    std::string input_img_dir_;
    std::string camera_extrinsic_path_;
    std::string camera_intrinsic_path_;
    std::string projective_z_buffer_dir_;
    std::string output_path_;
};

struct Observation {
    int camera_id;
    int keypoint_idx;

    Vec2d pixel;
    double depth = 0.0;  
};

struct Landmark {
    int landmark_id;

    Vec3d initial_position;
    Vec3d optimized_position;

    std::vector<Observation> observations;
};


struct Camera{
 int camera_id_;
 std::string camera_name_;
 std::string camera_path_;
 double time_stamp_;
 SE3d initial_T_wc_;
 // add depth or descriptor
 cv::Mat depth_map_;
 cv::Mat descriptors_;
 std::vector<cv::KeyPoint> keypoints_;
};