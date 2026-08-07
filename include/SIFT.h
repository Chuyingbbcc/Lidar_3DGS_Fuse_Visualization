#pragma once

#include "General.h"

#include <map>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>


using FeatureNode = std::pair<int, int>;
// FeatureNode = {camera_id, keypoint_index}


class SIFT
{
public:

    static Status extract_sift(
        Camera& camera,
        Config& config);

    static Status exhaust_pair_matching(
        const std::map<int, Camera>& camera_map,
        Config& config,
        std::map<int, Landmark>& landmarks);


private:

    //----------------------------------------------------------
    // Find 2 nearest descriptor matches
    //----------------------------------------------------------
    static void knn_matching(
        const cv::Mat& desc_1,
        const cv::Mat& desc_2,
        std::vector<std::vector<cv::DMatch>>& knn_matches);


    //----------------------------------------------------------
    // Lowe ratio test
    //----------------------------------------------------------
    static void lowe_ratio_test(
        const std::vector<std::vector<cv::DMatch>>& knn_matches,
        std::vector<cv::DMatch>& ratio_matches,
        float ratio_threshold);


    //----------------------------------------------------------
    // Convert DMatch -> pixel coordinates
    //----------------------------------------------------------
    static void match_to_pixels(
        const std::vector<cv::DMatch>& ratio_matches,
        const Camera& camera_1,
        const Camera& camera_2,
        std::vector<cv::Point2f>& points_1,
        std::vector<cv::Point2f>& points_2);


    //----------------------------------------------------------
    // Fundamental matrix + RANSAC
    //----------------------------------------------------------
    static int geometric_verification(
        const std::vector<cv::Point2f>& points_1,
        const std::vector<cv::Point2f>& points_2,
        double ransac_threshold,
        std::vector<uchar>& inlier_mask);


    //----------------------------------------------------------
    // Merge matched features into Union-Find
    //----------------------------------------------------------
    static void union_matches(
        int camera_id_1,
        int camera_id_2,
        const std::vector<cv::DMatch>& ratio_matches,
        const std::vector<uchar>& inlier_mask,
        std::map<FeatureNode, FeatureNode>& parent);


    //----------------------------------------------------------
    // Convert Union-Find groups to Landmark map
    //----------------------------------------------------------
    static void extract_landmark_map(
        const std::map<FeatureNode, FeatureNode>& parent,
        const std::map<int, Camera>& camera_map,
        std::map<int, Landmark>& landmark_map);
};