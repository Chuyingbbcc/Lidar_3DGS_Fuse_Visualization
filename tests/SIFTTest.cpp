//
// Created by chuchu on 8/6/26.
//
#include <gtest/gtest.h>

#include "SIFT.h"
#include "General.h"
#include <opencv2/opencv.hpp>
#include <iostream>

TEST(SIFTTest, ExtractSIFT)
{
    //----------------------------------------------------------
    // Load config
    //----------------------------------------------------------
    std::string config_path = std::string(PROJECT_ROOT_DIR)
    + "/src/config.yaml";
    std::cout << "Project root: "
          << PROJECT_ROOT_DIR
          << std::endl;

    std::cout << "Config path: "
          << config_path
          << std::endl;
   Config config(config_path);


    //----------------------------------------------------------
    // Build a camera
    //----------------------------------------------------------
    Camera camera;

    camera.camera_id_ = 0;
    camera.camera_name_ = "test";
    camera.camera_path_ = "/home/chuchu/undistorted/IMG_2644.png";

    //----------------------------------------------------------
    // Run SIFT
    //----------------------------------------------------------
    Status status = SIFT::extract_sift(camera, config);

    //----------------------------------------------------------
    // Verify
    //----------------------------------------------------------
    ASSERT_TRUE(status.success)
        << status.message;

    ASSERT_FALSE(camera.keypoints_.empty());
    ASSERT_FALSE(camera.descriptors_.empty());

    EXPECT_EQ(
        camera.keypoints_.size(),
        static_cast<size_t>(camera.descriptors_.rows)
    );

    EXPECT_EQ(
        128,
        camera.descriptors_.cols
    );

    EXPECT_EQ(
        CV_32F,
        camera.descriptors_.type()
    );

    //----------------------------------------------------------
    // Visualize SIFT keypoints
    //----------------------------------------------------------
    cv::Mat image = cv::imread(camera.camera_path_);

    ASSERT_FALSE(image.empty())
        << "Failed to load image: "
        << camera.camera_path_;

    cv::Mat visualization;

    cv::drawKeypoints(
        image,
        camera.keypoints_,
        visualization,
        cv::Scalar::all(-1),
        cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS
    );

    //----------------------------------------------------------
    // Save result
    //----------------------------------------------------------
    const std::string output_path =
       std::string(PROJECT_ROOT_DIR)
    + "/output/sift_result.png";

    bool saved = cv::imwrite(
        output_path,
        visualization
    );

    EXPECT_TRUE(saved);

    std::cout
        << "SIFT keypoints: "
        << camera.keypoints_.size()
        << std::endl;

    std::cout
        << "Visualization saved to: "
        << output_path
        << std::endl;

    //----------------------------------------------------------
    // Optional: display image
    //----------------------------------------------------------
    cv::imshow(
        "SIFT Keypoints",
        visualization
    );

    cv::waitKey(0);
}

