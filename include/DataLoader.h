//
// Created by chuchu on 8/5/26.
//
#pragma once
#include <map>
#include <General.h>

class DataLoader{
public:
 DataLoader(Config& config): config_(config){};
 Status load();
 std::map<int, Camera>& get_camera_map();
private:
 Config config_;
 std::map<int, Camera>camera_map_;
 void load_intrinsic();
 void load_extrinsic();
 void load_depth();
 void load_camera();

};
