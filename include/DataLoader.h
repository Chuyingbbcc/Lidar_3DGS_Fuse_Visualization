//
// Created by chuchu on 8/5/26.
//
#pragma once
#include <map>
#include <General.h>
#include "DataType.h"
#include <nlohmann/json.hpp>

class DataLoader{
public:
 DataLoader(Config& config): config_(config){};
 Status load();
 std::map<int, Camera>& get_camera_map();
private:
 Config config_;
 CameraIntrinsic intrinsic_;
 CameraExtrinsic extrinsic_;
 std::map<int, Camera>camera_map_;
 CameraIntrinsic load_intrinsic();
 CameraExtrinsic load_extrinsic();
 void load_depth();
 void load_camera();

//helper
 Mat4d load_matrix4d(
    const nlohmann::json& json_matrix);



};
