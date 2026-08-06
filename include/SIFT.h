//
// Created by chuchu on 8/5/26.
//
#include "General.h"
#include <map>
#pragma once

class SIFT{
public:

    static Status extract_sift( Camera& camera);

    static Status exhaust_pair_matching(
        const std::map<int, Camera>& camera_map,
        Config& config,std::map<int, Landmark>& landmarks);

    /*static bool geometricVerification(
        const std::vector<cv::KeyPoint>& keypoints1,
        const std::vector<cv::KeyPoint>& keypoints2,
        const std::vector<cv::DMatch>& matches,
        std::vector<cv::DMatch>& inlier_matches,
        Eigen::Matrix3d& fundamental_matrix);*/
};
