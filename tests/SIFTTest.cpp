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
    camera.camera_path_ = std::string(PROJECT_ROOT_DIR) + "/data/undistorted/IMG_2644.png";

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
    // Optional: display image (skipped when no display server is
    // available, e.g. headless CI, since cv::imshow aborts the
    // process rather than throwing in that case)
    //----------------------------------------------------------
    if (std::getenv("DISPLAY") != nullptr) {
        cv::imshow(
            "SIFT Keypoints",
            visualization
        );

        cv::waitKey(0);
    }
}


TEST(SIFTTest, MatchAndVisualizeTwoImages)
{
    const std::string config_path =
        std::string(PROJECT_ROOT_DIR) + "/src/config.yaml";
    Config config(config_path);

    Camera camera_1;
    camera_1.camera_id_ = 0;
    camera_1.camera_name_ = "IMG_2644.png";
    camera_1.camera_path_ =
        std::string(PROJECT_ROOT_DIR) + "/test_data/IMG_2128.png";

    Camera camera_2;
    camera_2.camera_id_ = 1;
    camera_2.camera_name_ = "IMG_2645.png";
    camera_2.camera_path_ =
        std::string(PROJECT_ROOT_DIR) + "/test_data/IMG_2136.png";

    ASSERT_TRUE(fs::exists(camera_1.camera_path_));
    ASSERT_TRUE(fs::exists(camera_2.camera_path_));

    const Status status_1 = SIFT::extract_sift(camera_1, config);
    const Status status_2 = SIFT::extract_sift(camera_2, config);

    ASSERT_TRUE(status_1.success) << status_1.message;
    ASSERT_TRUE(status_2.success) << status_2.message;
    ASSERT_FALSE(camera_1.descriptors_.empty());
    ASSERT_FALSE(camera_2.descriptors_.empty());

    std::vector<cv::DMatch> candidate_matches;
    std::vector<uchar> inlier_mask;

    SIFT::match_pair(
        camera_1.camera_id_,
        camera_1,
        camera_2.camera_id_,
        camera_2,
        static_cast<float>(config.ratio_threshold_),
        config.ransac_threshold_,
        config.min_inliers_,
        candidate_matches,
        inlier_mask,
        static_cast<float>(config.sift_max_match_distance_));

    ASSERT_FALSE(candidate_matches.empty())
        << "No geometrically verified SIFT matches were found.";
    ASSERT_EQ(candidate_matches.size(), inlier_mask.size());

    // Keep only RANSAC inliers so every line in the output represents a
    // geometrically verified feature correspondence.
    std::vector<cv::DMatch> inlier_matches;
    for (size_t i = 0; i < candidate_matches.size(); ++i)
    {
        if (inlier_mask[i])
        {
            inlier_matches.push_back(candidate_matches[i]);
        }
    }

    ASSERT_GE(
        inlier_matches.size(),
        static_cast<size_t>(config.min_inliers_));

    const cv::Mat image_1 = cv::imread(camera_1.camera_path_);
    const cv::Mat image_2 = cv::imread(camera_2.camera_path_);
    ASSERT_FALSE(image_1.empty());
    ASSERT_FALSE(image_2.empty());

    cv::Mat visualization;
    cv::drawMatches(
        image_1,
        camera_1.keypoints_,
        image_2,
        camera_2.keypoints_,
        inlier_matches,
        visualization,
        cv::Scalar::all(-1),
        cv::Scalar::all(-1),
        std::vector<char>(),
        cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    const fs::path output_dir =
        fs::path(PROJECT_ROOT_DIR) / "output";
    fs::create_directories(output_dir);

    const fs::path output_path =
        output_dir / "sift_matches_IMG_2644_IMG_2645.png";
    ASSERT_TRUE(cv::imwrite(output_path.string(), visualization));

    std::cout
        << "Verified SIFT matches: " << inlier_matches.size() << '\n'
        << "Match visualization saved to: " << output_path << std::endl;
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


TEST(SIFTTest, SequentialPairMatchingRejectsInvalidWindow)
{
    Config config;
    config.sequential_match_window_size_ = 0;

    std::map<int, Camera> camera_map;
    std::map<int, Landmark> landmark_map;

    const Status status = SIFT::sequential_pair_matching(
        camera_map,
        config,
        landmark_map);

    EXPECT_FALSE(status.success);
    EXPECT_NE(
        status.message.find("window_size"),
        std::string::npos);
    EXPECT_TRUE(landmark_map.empty());
}



TEST(SIFTTest, ExtractAndTrackIntegration)
{
    //----------------------------------------------------------
    // Load config
    //----------------------------------------------------------
    const std::string config_path =
        std::string(PROJECT_ROOT_DIR) + "/src/config.yaml";

    Config config(config_path);

    ASSERT_GT(config.sequential_match_window_size_, 0);

    //----------------------------------------------------------
    // Collect images from configured input directory
    //----------------------------------------------------------
    std::vector<fs::path> image_paths;
    fs::path input_img_dir(config.input_img_dir_);

    if (input_img_dir.is_relative())
    {
        input_img_dir =
            (fs::path(config_path).parent_path() / input_img_dir)
                .lexically_normal();
    }

    ASSERT_TRUE(fs::exists(input_img_dir))
        << "Image directory does not exist: "
        << input_img_dir;

    ASSERT_TRUE(fs::is_directory(input_img_dir))
        << "Input image path is not a directory: "
        << input_img_dir;

    for (const auto& entry :
         fs::directory_iterator(input_img_dir))
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
    constexpr size_t MAX_TEST_IMAGES = 20;

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
    // Diagnose image pairs for which geometric verification
    // rejects most of the descriptor matches.
    //----------------------------------------------------------
    constexpr double LOW_INLIER_RATIO = 0.25;
    constexpr size_t MAX_LOW_INLIER_VISUALIZATIONS = 10;

    const fs::path low_inlier_output_dir =
        fs::path(PROJECT_ROOT_DIR) / "output" / "sift_low_inlier_pairs";

    size_t low_inlier_pair_count = 0;
    size_t saved_low_inlier_pair_count = 0;

    for (auto camera_1_it = camera_map.begin();
         camera_1_it != camera_map.end();
         ++camera_1_it)
    {
        for (auto camera_2_it = std::next(camera_1_it);
             camera_2_it != camera_map.end();
             ++camera_2_it)
        {
            const Camera& camera_1 = camera_1_it->second;
            const Camera& camera_2 = camera_2_it->second;

            std::vector<std::vector<cv::DMatch>> knn_matches;
            SIFT::knn_matching(
                camera_1.descriptors_,
                camera_2.descriptors_,
                knn_matches);

            std::vector<cv::DMatch> ratio_matches;
            SIFT::lowe_ratio_test(
                knn_matches,
                ratio_matches,
                static_cast<float>(config.ratio_threshold_),
                static_cast<float>(config.sift_max_match_distance_));

            // Very small match sets are not meaningful for fundamental
            // matrix estimation and are already rejected by match_pair().
            if (ratio_matches.size() <
                static_cast<size_t>(config.min_inliers_))
            {
                continue;
            }

            std::vector<cv::Point2f> points_1;
            std::vector<cv::Point2f> points_2;
            SIFT::match_to_pixels(
                ratio_matches,
                camera_1,
                camera_2,
                points_1,
                points_2);

            std::vector<uchar> inlier_mask;
            const int num_inliers = SIFT::geometric_verification(
                points_1,
                points_2,
                config.ransac_threshold_,
                inlier_mask);

            const double inlier_ratio =
                static_cast<double>(num_inliers) /
                static_cast<double>(ratio_matches.size());

            if (inlier_ratio >= LOW_INLIER_RATIO)
            {
                continue;
            }

            ++low_inlier_pair_count;

            std::cout
                << "Low-inlier SIFT pair: "
                << camera_1.camera_name_ << " <-> "
                << camera_2.camera_name_ << ", ratio matches="
                << ratio_matches.size() << ", inliers="
                << num_inliers << ", retention="
                << 100.0 * inlier_ratio << "%" << std::endl;

            if (saved_low_inlier_pair_count >=
                MAX_LOW_INLIER_VISUALIZATIONS)
            {
                continue;
            }

            // geometric_verification() leaves the mask empty when no
            // fundamental matrix can be estimated. Treat every match as an
            // outlier in that case so the failed pair can still be drawn.
            if (inlier_mask.size() != ratio_matches.size())
            {
                inlier_mask.assign(ratio_matches.size(), 0);
            }

            std::vector<char> outlier_draw_mask(ratio_matches.size(), 0);
            std::vector<char> inlier_draw_mask(ratio_matches.size(), 0);
            for (size_t match_idx = 0;
                 match_idx < ratio_matches.size();
                 ++match_idx)
            {
                inlier_draw_mask[match_idx] = inlier_mask[match_idx] != 0;
                outlier_draw_mask[match_idx] = inlier_mask[match_idx] == 0;
            }

            const cv::Mat image_1 = cv::imread(camera_1.camera_path_);
            const cv::Mat image_2 = cv::imread(camera_2.camera_path_);
            ASSERT_FALSE(image_1.empty());
            ASSERT_FALSE(image_2.empty());

            cv::Mat visualization;

            // Red: passed descriptor matching but rejected by RANSAC.
            cv::drawMatches(
                image_1,
                camera_1.keypoints_,
                image_2,
                camera_2.keypoints_,
                ratio_matches,
                visualization,
                cv::Scalar(0, 0, 255),
                cv::Scalar(0, 0, 255),
                outlier_draw_mask,
                cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

            // Green: geometrically verified RANSAC inliers.
            cv::drawMatches(
                image_1,
                camera_1.keypoints_,
                image_2,
                camera_2.keypoints_,
                ratio_matches,
                visualization,
                cv::Scalar(0, 255, 0),
                cv::Scalar(0, 255, 0),
                inlier_draw_mask,
                cv::DrawMatchesFlags::DRAW_OVER_OUTIMG |
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

            const std::string diagnostic_text =
                "ratio matches: " + std::to_string(ratio_matches.size()) +
                "  inliers: " + std::to_string(num_inliers) +
                "  retention: " +
                std::to_string(100.0 * inlier_ratio) + "%";

            cv::rectangle(
                visualization,
                cv::Rect(0, 0, 850, 48),
                cv::Scalar(0, 0, 0),
                cv::FILLED);
            cv::putText(
                visualization,
                diagnostic_text,
                cv::Point(12, 33),
                cv::FONT_HERSHEY_SIMPLEX,
                0.8,
                cv::Scalar(255, 255, 255),
                2);

            fs::create_directories(low_inlier_output_dir);
            const fs::path output_path =
                low_inlier_output_dir /
                (camera_1_it->second.camera_name_ + "_vs_" +
                 camera_2_it->second.camera_name_ + ".jpg");

            ASSERT_TRUE(cv::imwrite(
                output_path.string(),
                visualization,
                {cv::IMWRITE_JPEG_QUALITY, 90}));

            ++saved_low_inlier_pair_count;
            std::cout << "Saved low-inlier visualization: "
                      << output_path << std::endl;
        }
    }

    std::cout
        << "Low-inlier pairs (< " << 100.0 * LOW_INLIER_RATIO
        << "% retention): " << low_inlier_pair_count
        << "; visualizations saved: "
        << saved_low_inlier_pair_count << std::endl;

    //----------------------------------------------------------
    // Run matching + track construction + landmark extraction
    //----------------------------------------------------------
    std::map<int, Landmark> landmark_map;

    Status status =
        SIFT::sequential_pair_matching(
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
   //----------------------------------------------------------
// Landmark track-length statistics
//----------------------------------------------------------
std::map<size_t, size_t> track_length_histogram;

size_t total_observations = 0;

for (const auto& [landmark_id, landmark] : landmark_map)
{
    const size_t track_length =
        landmark.observations_.size();

    ++track_length_histogram[track_length];

    total_observations += track_length;
}

std::cout << "\n";
std::cout << "========================================\n";
std::cout << " SIFT Track Statistics\n";
std::cout << "========================================\n";

std::cout
    << "Images: "
    << num_images
    << "\n";

std::cout
    << "Landmarks: "
    << landmark_map.size()
    << "\n\n";

for (const auto& [track_length, count] :
     track_length_histogram)
{
    const double percentage =
        landmark_map.empty()
            ? 0.0
            : 100.0
                * static_cast<double>(count)
                / static_cast<double>(
                    landmark_map.size());

    std::cout
        << "Track length "
        << track_length
        << " : "
        << count
        << " landmarks"
        << " ("
        << percentage
        << "%)"
        << std::endl;
}

if (!landmark_map.empty())
{
    const double average_track_length =
        static_cast<double>(total_observations)
        / static_cast<double>(
            landmark_map.size());

    std::cout
        << "\nAverage track length: "
        << average_track_length
        << std::endl;
}

std::cout
    << "Total observations: "
    << total_observations
    << std::endl;

std::cout << "========================================\n";
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
    const size_t track_length =
        landmark.observations_.size();

    cv::Scalar color;

    if (track_length == 2)
    {
        // Red
        color = cv::Scalar(0, 0, 255);
    }
    else if (track_length == 3)
    {
        // Yellow
        color = cv::Scalar(0, 255, 255);
    }
    else
    {
        // Green
        color = cv::Scalar(0, 255, 0);
    }

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
        // Draw observation
        //--------------------------------------------------
        cv::circle(
            image,
            point,
            4,
            color,
            2);

        //--------------------------------------------------
        // Draw landmark ID + track length
        //
        // Example:
        // 125(2)
        // means landmark 125 has 2 observations.
        //--------------------------------------------------
        std::string label =
            std::to_string(landmark_id)
            + "("
            + std::to_string(track_length)
            + ")";

        cv::putText(
            image,
            label,
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


TEST(SIFTTest, SequentialPairMatchingVisualizationIntegration)
{
    const std::string config_path =
        std::string(PROJECT_ROOT_DIR) + "/src/config.yaml";
    Config config(config_path);

    ASSERT_GT(config.sequential_match_window_size_, 0);

    fs::path input_img_dir(config.input_img_dir_);
    if (input_img_dir.is_relative())
    {
        input_img_dir =
            (fs::path(config_path).parent_path() / input_img_dir)
                .lexically_normal();
    }

    ASSERT_TRUE(fs::is_directory(input_img_dir));

    std::vector<fs::path> image_paths;
    for (const auto& entry : fs::directory_iterator(input_img_dir))
    {
        if (!entry.is_regular_file())
        {
            continue;
        }

        std::string extension = entry.path().extension().string();
        std::transform(
            extension.begin(),
            extension.end(),
            extension.begin(),
            [](unsigned char value)
            {
                return static_cast<char>(std::tolower(value));
            });

        if (extension == ".png" || extension == ".jpg" ||
            extension == ".jpeg" || extension == ".bmp")
        {
            image_paths.push_back(entry.path());
        }
    }

    std::sort(image_paths.begin(), image_paths.end());
    ASSERT_GE(image_paths.size(), 2u);

    // A short sequence keeps this integration test fast while still testing
    // the beginning, middle, and truncated end of the matching window.
    constexpr size_t MAX_TEST_IMAGES = 6;
    const size_t num_images =
        std::min(image_paths.size(), MAX_TEST_IMAGES);

    std::map<int, Camera> camera_map;
    for (size_t i = 0; i < num_images; ++i)
    {
        Camera camera;
        camera.camera_id_ = static_cast<int>(i);
        camera.camera_name_ = image_paths[i].filename().string();
        camera.camera_path_ = image_paths[i].string();

        const Status extract_status = SIFT::extract_sift(camera, config);
        ASSERT_TRUE(extract_status.success) << extract_status.message;
        ASSERT_FALSE(camera.descriptors_.empty());

        camera_map.emplace(camera.camera_id_, std::move(camera));
    }

    std::map<int, Landmark> landmarks;
    const Status match_status = SIFT::sequential_pair_matching(
        camera_map,
        config,
        landmarks);

    ASSERT_TRUE(match_status.success) << match_status.message;
    ASSERT_FALSE(landmarks.empty());

    const fs::path output_dir =
        fs::path(PROJECT_ROOT_DIR) /
        "output" /
        "sequential_pair_matches";
    fs::create_directories(output_dir);

    size_t visualization_count = 0;

    for (size_t i = 0; i < num_images; ++i)
    {
        const size_t window_end = std::min(
            num_images,
            i + static_cast<size_t>(
                    config.sequential_match_window_size_) + 1);

        for (size_t j = i + 1; j < window_end; ++j)
        {
            const Camera& camera_1 =
                camera_map.at(static_cast<int>(i));
            const Camera& camera_2 =
                camera_map.at(static_cast<int>(j));

            std::vector<cv::DMatch> ratio_matches;
            std::vector<uchar> inlier_mask;
            SIFT::match_pair(
                camera_1.camera_id_,
                camera_1,
                camera_2.camera_id_,
                camera_2,
                static_cast<float>(config.ratio_threshold_),
                config.ransac_threshold_,
                config.min_inliers_,
                ratio_matches,
                inlier_mask,
                static_cast<float>(config.sift_max_match_distance_));

            // Draw only the direct RANSAC inliers. Landmark tracks may also
            // connect two cameras transitively through intermediate frames.
            std::vector<cv::DMatch> pair_matches;
            for (size_t match_idx = 0;
                 match_idx < ratio_matches.size();
                 ++match_idx)
            {
                if (inlier_mask[match_idx])
                {
                    pair_matches.push_back(ratio_matches[match_idx]);
                }
            }

            if (pair_matches.empty())
            {
                continue;
            }

            const cv::Mat image_1 = cv::imread(camera_1.camera_path_);
            const cv::Mat image_2 = cv::imread(camera_2.camera_path_);
            ASSERT_FALSE(image_1.empty());
            ASSERT_FALSE(image_2.empty());

            cv::Mat visualization;
            cv::drawMatches(
                image_1,
                camera_1.keypoints_,
                image_2,
                camera_2.keypoints_,
                pair_matches,
                visualization,
                cv::Scalar(0, 255, 0),
                cv::Scalar(0, 255, 0),
                std::vector<char>(),
                cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

            const std::string label =
                camera_1.camera_name_ + " <-> " +
                camera_2.camera_name_ + " | verified matches: " +
                std::to_string(pair_matches.size());
            cv::rectangle(
                visualization,
                cv::Rect(0, 0, 1000, 48),
                cv::Scalar(0, 0, 0),
                cv::FILLED);
            cv::putText(
                visualization,
                label,
                cv::Point(12, 33),
                cv::FONT_HERSHEY_SIMPLEX,
                0.75,
                cv::Scalar(255, 255, 255),
                2);

            const fs::path output_path =
                output_dir /
                (camera_1.camera_name_ + "_vs_" +
                 camera_2.camera_name_ + ".jpg");

            ASSERT_TRUE(cv::imwrite(
                output_path.string(),
                visualization,
                {cv::IMWRITE_JPEG_QUALITY, 90}));

            ++visualization_count;
            std::cout
                << "Saved sequential match visualization: "
                << output_path << std::endl;
        }
    }

    EXPECT_GT(visualization_count, 0u);
    std::cout
        << "Sequential matching generated " << landmarks.size()
        << " landmark tracks and " << visualization_count
        << " pair visualizations." << std::endl;
}
