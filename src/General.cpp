//
// Created by chuchu on 8/5/26.
//
#include "General.h"

#include <yaml-cpp/yaml.h>

Config::Config(const std::string& yaml_path)
{
    YAML::Node config = YAML::LoadFile(yaml_path);

    // Paths
    input_img_dir_ = config["input_img_dir"].as<std::string>();
    camera_extrinsic_path_ = config["camera_extrinsic_path"].as<std::string>();
    camera_intrinsic_path_ = config["camera_intrinsic_path"].as<std::string>();
    camera_timestamp_path_ = config["camera_timestamp_path"].as<std::string>();
    projective_z_buffer_dir_ = config["projective_z_buffer_dir"].as<std::string>();
    output_path_ = config["output_path"].as<std::string>();
    lidar_ply_dirs_ = config["lidar_ply_dirs"].as<std::string>();
    lidar_kf_path_ = config["lidar_kf_path"].as<std::string>();

    // SIFT
    sift_nfeatures_ =
        config["sift_nfeatures"].as<int>();

    sift_n_octave_layers_ =
        config["sift_n_octave_layers"].as<int>();

    sift_contrast_threshold_ =
        config["sift_contrast_threshold"].as<double>();

    sift_edge_threshold_ =
        config["sift_edge_threshold"].as<double>();

    sift_sigma_ =
        config["sift_sigma"].as<double>();

    // Feature matching
    ratio_threshold_ =
        config["ratio_threshold"].as<double>();

    ransac_threshold_ =
        config["ransac_threshold"].as<double>();

    min_inliers_ =
        config["min_inliers"].as<int>();
}