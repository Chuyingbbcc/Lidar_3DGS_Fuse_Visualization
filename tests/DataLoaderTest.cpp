#include "DataLoader.h"
#include <iostream>
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <filesystem>



TEST(DataLoaderTest, Loading_test) {
    Config config("../src/config.yaml");
    DataLoader data_loader(config);
    data_loader.load();

    auto& lidar_info_map = data_loader.get_lidar_info_map();

    // Check if the lidar_info_map_ is not empty
    EXPECT_FALSE(lidar_info_map.empty());

    auto& camera_map = data_loader.get_camera_map();

    // Check if the camera_map_ is not empty
    EXPECT_FALSE(camera_map.empty());
    // print out camera IDs
    for (const auto& pair : camera_map) {
        std::cout << "Camera ID: " << pair.first<<" "<<pair.second.camera_id_<<" "<<pair.second.camera_path_ << std::endl;
    }

    for(auto& [camera_id, camera] : data_loader.get_camera_map()) {
        std::cout << "Camera ID: " << camera_id << " associated Lidar: " <<camera.matched_lidar_id_ << std::endl;
    }

    // Save each camera's depth map and visualize one of them.
    const std::filesystem::path depth_map_dir = "../data/depth_maps";
    std::filesystem::create_directories(depth_map_dir);

    int first_valid_camera_id = -1;
    for (auto& [camera_id, camera] : camera_map) {
        if (camera.depth_map_.empty())
            continue;

        // Raw float depth (meters) for downstream processing.
        const std::filesystem::path raw_path = depth_map_dir / (camera.camera_name_ + "_depth.tiff");
        cv::imwrite(raw_path.string(), camera.depth_map_);

        // 8-bit normalized visualization for quick inspection.
        cv::Mat depth_vis;
        cv::normalize(camera.depth_map_, depth_vis, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        cv::Mat depth_color;
        cv::applyColorMap(depth_vis, depth_color, cv::COLORMAP_JET);
        const std::filesystem::path vis_path = depth_map_dir / (camera.camera_name_ + "_depth_vis.png");
        cv::imwrite(vis_path.string(), depth_color);

        if (first_valid_camera_id == -1)
            first_valid_camera_id = camera_id;
    }

    // View one of the saved depth maps.
    // if (first_valid_camera_id != -1) {
    //     const auto& camera = camera_map.at(first_valid_camera_id);
    //     cv::Mat depth_vis;
    //     cv::normalize(camera.depth_map_, depth_vis, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    //     cv::Mat depth_color;
    //     cv::applyColorMap(depth_vis, depth_color, cv::COLORMAP_JET);
    //     cv::imshow("Depth map - camera " + std::to_string(first_valid_camera_id), depth_color);
    //     cv::waitKey(0);
    //     cv::destroyAllWindows();
    // }
}

