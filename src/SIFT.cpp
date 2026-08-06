//
// Created by chuchu on 8/5/26.
//


#include "General.h"
#include "SIFT.h"
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>


using namespace std;
Status SIFT::extract_sift( Camera& camera,Config& config){
    // Load image.
    cv::Mat image = cv::imread(
        camera.camera_path_,
        cv::IMREAD_GRAYSCALE);

    if (image.empty())
    {
        return {
            false,
            "Failed to load image: " + camera.camera_path_
        };
    }

    // Create SIFT detector.
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create(
        config.sift_nfeatures_,
        config.sift_n_octave_layers_,
        config.sift_contrast_threshold_,
        config.sift_edge_threshold_,
        config.sift_sigma_);

    // Clear previous extraction results.
    camera.keypoints_.clear();
    camera.descriptors_.release();

    // Detect keypoints and compute descriptors.
    sift->detectAndCompute(
        image,
        cv::noArray(),
        camera.keypoints_,
        camera.descriptors_);

    if (camera.keypoints_.empty())
    {
        return {
            false,
            "No SIFT features detected in image: "
                + camera.camera_name_
        };
    }

    if (camera.descriptors_.empty())
    {
        return {
            false,
            "Failed to compute SIFT descriptors for image: "
                + camera.camera_name_
        };
    }

    return {
        true,
        "Extracted "
            + std::to_string(camera.keypoints_.size())
            + " SIFT features from "
            + camera.camera_name_
    };
}

Status SIFT::exhaust_pair_matching(
    const std::map<int, Camera>& camera_map,
    Config& config,std::map<int, Landmark>& landmarks){
    //Todo:: add implementation
    return {true, "Match OK!"};
}
