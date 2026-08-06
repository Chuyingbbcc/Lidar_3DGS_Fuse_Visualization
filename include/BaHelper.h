//
// Created by chuchu on 8/5/26.
//

#pragma once
#include "General.h"
#include "DataType.h"
#include <map>


class BaHelper{
public:
static Status  load_projected_depth(Camera& camera);
static Status  extract_initial_landmark_world_pos(std::map<int ,Camera>& camera_map, Config& config,  std::map<int, Landmark>&landmarks );
static Status  writeOptimziedCamera(std::map<int ,Camera>& camera_map, Config& config);
};