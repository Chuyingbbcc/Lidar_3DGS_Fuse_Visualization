#pragma once
#include <map>
#include <string>
#include "General.h"

class BundleAdjustmentOptimizer
{
public:
    BundleAdjustmentOptimizer() = default;
    ~BundleAdjustmentOptimizer() = default;

    // Input data
    void SetCameraMap(const std::map<int, Camera>& camera_map);

    void SetLandmarkMap(const std::map<int, Landmark>& landmark_map);

    // Run bundle adjustment
    Status Optimize();

    // Optional helper if caller only wants optimized poses
    void GetOptimizedPoses(
        std::map<int, SE3d>& optimized_poses) const;


};