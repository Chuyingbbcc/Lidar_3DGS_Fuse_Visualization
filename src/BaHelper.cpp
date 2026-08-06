//
// Created by chuchu on 8/5/26.
//

#include "BaHelper.h"
using namespace std;
Status BaHelper::load_projected_depth(Camera& camera){
return {true, "depth load ok!"};
}

Status  BaHelper::extract_initial_landmark_world_pos(std::map<int ,Camera>& camera_map, Config& config,  std::map<int, Landmark>&landmarks ){
return {true, "initial landmark ok!"};
}


Status  BaHelper::writeOptimziedCamera(std::map<int ,Camera>& camera_map, Config& config){
  return {true, "write done"};
}








