//
// Created by chuchu on 8/5/26.
//
#include "Optimizer.h"
#include <map>

void BundleAdjustmentOptimizer::SetCameraMap(const std::map<int, Camera>& camera_map){
 return;
}

void BundleAdjustmentOptimizer::SetLandmarkMap(const std::map<int, Landmark>& landmark_map){
return;
}

    // Run bundle adjustment
Status BundleAdjustmentOptimizer::Optimize(){
    return {true, "Optimize Ok"};
}

    // Optional helper if caller only wants optimized poses
void BundleAdjustmentOptimizer::GetOptimizedPoses(
        std::map<int, SE3d>& optimized_poses) const{
return;
}

