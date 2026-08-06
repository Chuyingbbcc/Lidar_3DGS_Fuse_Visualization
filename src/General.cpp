//
// Created by chuchu on 8/5/26.
//
#include "General.h"

#include <yaml-cpp/yaml.h>

Config::Config(const std::string& yaml_path)
{
    YAML::Node config = YAML::LoadFile(yaml_path);

    input_img_dir_ = config["input_img_dir"].as<std::string>();
    camera_extrinsic_path_ = config["camera_extrinsic_path"].as<std::string>();
    camera_intrinsic_path_ = config["camera_intrinsic_path"].as<std::string>();
    projective_z_buffer_dir_ = config["projective_z_buffer_dir"].as<std::string>();
    output_path_= config["output_path"].as<std::string>();

}