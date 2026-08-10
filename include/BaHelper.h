//
// Created by chuchu on 8/5/26.
//

#pragma once
#include "General.h"
#include "DataType.h"
#include <map>


class BaHelper{
public:
static Vec3d  PixelToCamera(const Vec2d& pixel,double depth,const Mat3d& K);
static Status  load_projected_depth(Camera& camera);
static Status  extract_landmark_world_pos(const std::map<int ,Camera>& camera_map, const CameraIntrinsic& intrinsic, const Config& config_, std::map<int, Landmark>&landmarks,  bool is_initial);
static Status  update_observation_depth(const std::map<int ,Camera>& camera_map, std::map<int, Landmark>& landmarks, bool is_initial);
static Status  writeOptimziedCamera(std::map<int ,Camera>& camera_map, Config& config);
// Writes camera poses (TUM format: timestamp tx ty tz qx qy qz qw, world<-camera) for trajectory visualization.
static Status  save_trajectory(const std::map<int ,Camera>& camera_map, const std::string& output_path, bool use_optimized);
};