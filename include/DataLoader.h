//
// Created by chuchu on 8/5/26.
//
#pragma once
#include <map>
#include "General.h"
#include "DataType.h"
#include <nlohmann/json.hpp>

class DataLoader{
public:
 DataLoader(Config& config): config_(config){};
 Status load();
 std::map<int, Camera>& get_camera_map();
 std::map<int, LidarPointCloudInfo>& get_lidar_info_map();
 void update_depth_map(const bool optimized);
 void update_depth_map_parallel(const bool optimized);
private:
 Config config_;
 CameraIntrinsic intrinsic_;
 CameraExtrinsic extrinsic_;
 std::map<int, Camera>camera_map_;
 std::map<int, LidarPointCloudInfo>lidar_info_map_;
 void load_intrinsic();
 void load_extrinsic();
 void load_depth();
 void load_camera();
 void load_lidar();
//helper
 Mat4d load_matrix4d(const nlohmann::json& json_matrix);
 void read_ply_xyz(const std::string& filename , std::vector<Vec3d>& points);
 void load_keyframe_jsonl();
 void camera_lidar_association();

};
