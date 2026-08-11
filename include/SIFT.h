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

    //----------------------------------------------------------
    // Match each camera with the next configured number of
    // cameras in map order.
    //----------------------------------------------------------
    static Status sequential_pair_matching(
        const std::map<int, Camera>& camera_map,
        Config& config,
        std::map<int, Landmark>& landmarks);

    //----------------------------------------------------------
    // Same as exhaust_pair_matching, but matches camera pairs
    // in parallel via tbb.
    //----------------------------------------------------------
    static Status exhaust_pair_matching_parallel(
        const std::map<int, Camera>& camera_map,
        Config& config,
        std::map<int, Landmark>& landmarks);


    //----------------------------------------------------------
    // Cache exhaust_pair_matching() results so a rerun can resume
    // without redoing the expensive exhaustive matching.
    //----------------------------------------------------------
    static Status save_landmarks(
        const std::string& file_path,
        const std::map<int, Landmark>& landmarks);

    static Status load_landmarks(
        const std::string& file_path,
        std::map<int, Landmark>& landmarks);


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
        float ratio_threshold,
        float max_match_distance = -1.0f);


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

    //----------------------------------------------------------
    // Match + verify a single camera pair. Thread-safe: only
    // reads camera_map, writes nothing shared.
    //----------------------------------------------------------
    static void match_pair(
        int camera_id_1,
        const Camera& camera_1,
        int camera_id_2,
        const Camera& camera_2,
        float ratio_threshold,
        double ransac_threshold,
        int min_inliers,
        std::vector<cv::DMatch>& verified_matches,
        std::vector<uchar>& inlier_mask,
        float max_match_distance = -1.0f);
};
