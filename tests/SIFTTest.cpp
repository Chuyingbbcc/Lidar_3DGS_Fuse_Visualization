//
// Created by chuchu on 8/6/26.
//
#include <gtest/gtest.h>


#include "SIFT.h"
#include "DataType.h"
#include "General.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <map>
#include <filesystem>
#include <algorithm>
#include <vector>
#include <set>
#include <cctype>
#include <cmath>

namespace fs = std::filesystem;

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


TEST(SIFTTest, ExtractLandmarkMapBasic){
 // ============================================================
    // 1. Create three cameras with synthetic keypoints
    // ============================================================

    std::map<int, Camera> camera_map;

    Camera camera_0;
    camera_0.camera_id_ = 0;
    camera_0.keypoints_.push_back(
        cv::KeyPoint(cv::Point2f(100.0f, 200.0f), 1.0f));

    Camera camera_1;
    camera_1.camera_id_ = 1;
    camera_1.keypoints_.push_back(
        cv::KeyPoint(cv::Point2f(110.0f, 205.0f), 1.0f));

    Camera camera_2;
    camera_2.camera_id_ = 2;
    camera_2.keypoints_.push_back(
        cv::KeyPoint(cv::Point2f(120.0f, 210.0f), 1.0f));

    camera_map.emplace(0, camera_0);
    camera_map.emplace(1, camera_1);
    camera_map.emplace(2, camera_2);


    // ============================================================
    // 2. Create synthetic Union-Find structure
    //
    // These three features represent the SAME physical landmark:
    //
    // camera 0, keypoint 0
    // camera 1, keypoint 0
    // camera 2, keypoint 0
    //
    //            (0,0)
    //            /   \
    //         (1,0)  (2,0)
    //
    // ============================================================

    FeatureNode node_0{0, 0};
    FeatureNode node_1{1, 0};
    FeatureNode node_2{2, 0};

    std::map<FeatureNode, FeatureNode> parent;

    parent[node_0] = node_0;   // root
    parent[node_1] = node_0;
    parent[node_2] = node_0;


    // ============================================================
    // 3. Run
    // ============================================================

    std::map<int, Landmark> landmark_map;

    SIFT::extract_landmark_map(
        parent,
        camera_map,
        landmark_map);


    // ============================================================
    // 4. Verify
    // ============================================================

    ASSERT_EQ(landmark_map.size(), 1);

    const Landmark& landmark = landmark_map.at(0);

    EXPECT_EQ(landmark.landmark_id_, 0);

    ASSERT_EQ(landmark.observations_.size(), 3);


    // ------------------------------------------------------------
    // Observation from camera 0
    // ------------------------------------------------------------

    EXPECT_EQ(
        landmark.observations_[0].camera_id_,
        0);

    EXPECT_EQ(
        landmark.observations_[0].keypoint_idx_,
        0);

    EXPECT_DOUBLE_EQ(
        landmark.observations_[0].pixel_.x(),
        100.0);

    EXPECT_DOUBLE_EQ(
        landmark.observations_[0].pixel_.y(),
        200.0);


    // ------------------------------------------------------------
    // Observation from camera 1
    // ------------------------------------------------------------

    EXPECT_EQ(
        landmark.observations_[1].camera_id_,
        1);

    EXPECT_DOUBLE_EQ(
        landmark.observations_[1].pixel_.x(),
        110.0);

    EXPECT_DOUBLE_EQ(
        landmark.observations_[1].pixel_.y(),
        205.0);


    // ------------------------------------------------------------
    // Observation from camera 2
    // ------------------------------------------------------------

    EXPECT_EQ(
        landmark.observations_[2].camera_id_,
        2);

    EXPECT_DOUBLE_EQ(
        landmark.observations_[2].pixel_.x(),
        120.0);

    EXPECT_DOUBLE_EQ(
        landmark.observations_[2].pixel_.y(),
        210.0);
}



TEST(SIFTTest, ExtractAndTrackIntegration)
{
    //----------------------------------------------------------
    // Load config
    //----------------------------------------------------------
    const std::string config_path =
        std::string(PROJECT_ROOT_DIR) + "/src/config.yaml";

    Config config(config_path);

    //----------------------------------------------------------
    // Collect images from configured input directory
    //----------------------------------------------------------
    std::vector<fs::path> image_paths;

    ASSERT_TRUE(fs::exists(config.input_img_dir_))
        << "Image directory does not exist: "
        << config.input_img_dir_;

    ASSERT_TRUE(fs::is_directory(config.input_img_dir_))
        << "Input image path is not a directory: "
        << config.input_img_dir_;

    for (const auto& entry :
         fs::directory_iterator(config.input_img_dir_))
    {
        if (!entry.is_regular_file())
        {
            continue;
        }

        std::string extension =
            entry.path().extension().string();

        std::transform(
            extension.begin(),
            extension.end(),
            extension.begin(),
            [](unsigned char c)
            {
                return static_cast<char>(std::tolower(c));
            });

        if (extension == ".jpg" ||
            extension == ".jpeg" ||
            extension == ".png" ||
            extension == ".bmp")
        {
            image_paths.push_back(entry.path());
        }
    }

    //----------------------------------------------------------
    // Make test deterministic
    //----------------------------------------------------------
    std::sort(
        image_paths.begin(),
        image_paths.end());

    ASSERT_GE(image_paths.size(), 2)
        << "Need at least two images for feature matching.";

    //----------------------------------------------------------
    // Keep integration test reasonably fast.
    // Increase this if you want to test longer tracks.
    //----------------------------------------------------------
    constexpr size_t MAX_TEST_IMAGES = 10;

    const size_t num_images =
        std::min(image_paths.size(), MAX_TEST_IMAGES);

    //----------------------------------------------------------
    // Extract SIFT for each selected image
    //----------------------------------------------------------
    std::map<int, Camera> camera_map;

    for (size_t i = 0; i < num_images; ++i)
    {
        Camera camera;

        camera.camera_id_ = static_cast<int>(i);
        camera.camera_name_ =
            image_paths[i].filename().string();
        camera.camera_path_ =
            image_paths[i].string();

        Status status =
            SIFT::extract_sift(camera, config);

        ASSERT_TRUE(status.success)
            << "SIFT extraction failed for "
            << camera.camera_path_
            << ": "
            << status.message;

        ASSERT_FALSE(camera.keypoints_.empty());
        ASSERT_FALSE(camera.descriptors_.empty());

        EXPECT_EQ(
            camera.keypoints_.size(),
            static_cast<size_t>(
                camera.descriptors_.rows));

        camera_map.emplace(
            camera.camera_id_,
            std::move(camera));
    }

    ASSERT_EQ(camera_map.size(), num_images);

    //----------------------------------------------------------
    // Run matching + track construction + landmark extraction
    //----------------------------------------------------------
    std::map<int, Landmark> landmark_map;

    Status status =
        SIFT::exhaust_pair_matching(
            camera_map,
            config,
            landmark_map);

    ASSERT_TRUE(status.success)
        << status.message;

    //----------------------------------------------------------
    // The selected neighboring images should generate matches.
    //----------------------------------------------------------
    ASSERT_FALSE(landmark_map.empty())
        << "No landmarks were generated from "
        << num_images
        << " images.";

    //----------------------------------------------------------
    // Validate landmark / track structure
    //----------------------------------------------------------
    size_t multi_view_track_count = 0;

    for (const auto& [landmark_id, landmark] : landmark_map)
    {
        EXPECT_EQ(
            landmark.landmark_id_,
            landmark_id);

        ASSERT_GE(
            landmark.observations_.size(),
            2u);

        std::set<int> observed_camera_ids;

        for (const Observation& observation :
             landmark.observations_)
        {
            //--------------------------------------------------
            // Observation must reference an existing camera
            //--------------------------------------------------
            auto camera_it =
                camera_map.find(
                    observation.camera_id_);

            ASSERT_NE(
                camera_it,
                camera_map.end());

            const Camera& camera =
                camera_it->second;

            //--------------------------------------------------
            // Keypoint index must be valid
            //--------------------------------------------------
            ASSERT_GE(
                observation.keypoint_idx_,
                0);

            ASSERT_LT(
                observation.keypoint_idx_,
                static_cast<int>(
                    camera.keypoints_.size()));

            //--------------------------------------------------
            // Stored observation pixel must match keypoint
            //--------------------------------------------------
            const cv::Point2f& keypoint =
                camera.keypoints_[
                    observation.keypoint_idx_].pt;

            EXPECT_NEAR(
                observation.pixel_.x(),
                static_cast<double>(keypoint.x),
                1e-6);

            EXPECT_NEAR(
                observation.pixel_.y(),
                static_cast<double>(keypoint.y),
                1e-6);

            //--------------------------------------------------
            // A landmark may occur at most once per camera
            //--------------------------------------------------
            EXPECT_TRUE(
                observed_camera_ids.insert(
                    observation.camera_id_).second);
        }

        if (landmark.observations_.size() >= 3)
        {
            ++multi_view_track_count;
        }
    }

    //----------------------------------------------------------
    // Informational output. Do not require a 3-view track here,
    // because that depends strongly on the particular dataset.
    //----------------------------------------------------------
    std::cout
        << "Integration test images: "
        << num_images
        << std::endl;

    std::cout
        << "Generated landmarks: "
        << landmark_map.size()
        << std::endl;

    std::cout
        << "Tracks with >= 3 observations: "
        << multi_view_track_count
        << std::endl;

    //----------------------------------------------------------
// Visualize landmark tracks
//----------------------------------------------------------
const fs::path output_dir =
    fs::path(PROJECT_ROOT_DIR)
    / "output"
    / "sift_tracks";

fs::create_directories(output_dir);


//----------------------------------------------------------
// Make one visualization image per camera
//----------------------------------------------------------
std::map<int, cv::Mat> visualization_map;

for (const auto& [camera_id, camera] : camera_map)
{
    cv::Mat image =
        cv::imread(camera.camera_path_);

    ASSERT_FALSE(image.empty())
        << "Failed to load image for visualization: "
        << camera.camera_path_;

    visualization_map[camera_id] =
        image.clone();
}


//----------------------------------------------------------
// Draw each landmark
//----------------------------------------------------------
for (const auto& [landmark_id, landmark] : landmark_map)
{
    if (landmark.observations_.size() < 3)
   {
    continue;
   }
    //------------------------------------------------------
    // Deterministic color for this landmark.
    // Same landmark gets same color in every image.
    //------------------------------------------------------
    cv::RNG rng(
        static_cast<uint64_t>(landmark_id + 1) * 12345);

    cv::Scalar color(
        rng.uniform(50, 255),
        rng.uniform(50, 255),
        rng.uniform(50, 255));


    for (const Observation& observation :
         landmark.observations_)
    {
        auto vis_it =
            visualization_map.find(
                observation.camera_id_);

        if (vis_it == visualization_map.end())
        {
            continue;
        }

        cv::Mat& image =
            vis_it->second;

        cv::Point point(
            static_cast<int>(
                std::round(
                    observation.pixel_.x())),
            static_cast<int>(
                std::round(
                    observation.pixel_.y())));


        //--------------------------------------------------
        // Draw feature location
        //--------------------------------------------------
        cv::circle(
            image,
            point,
            4,
            color,
            2);


        //--------------------------------------------------
        // Draw landmark id
        //--------------------------------------------------
        cv::putText(
            image,
            std::to_string(landmark_id),
            point + cv::Point(5, -5),
            cv::FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1);
    }
}


//----------------------------------------------------------
// Save visualization images
//----------------------------------------------------------
for (const auto& [camera_id, image] :
     visualization_map)
{
    const Camera& camera =
        camera_map.at(camera_id);

    const fs::path output_path =
        output_dir /
        (
            "tracks_" +
            camera.camera_name_
        );

    const bool saved =
        cv::imwrite(
            output_path.string(),
            image);

    EXPECT_TRUE(saved);

    std::cout
        << "Saved track visualization: "
        << output_path
        << std::endl;
   }
}

