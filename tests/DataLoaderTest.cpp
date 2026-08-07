#include "DataLoader.h"
#include <iostream>
#include <gtest/gtest.h>

TEST(DataLoaderTest, LoadLidar) {
    Config config("../src/config.yaml");
    DataLoader data_loader(config);
    auto& lidar_info_map = data_loader.get_lidar_info_map();

    // Check if the lidar_info_map_ is not empty
    EXPECT_FALSE(lidar_info_map.empty());

    // check if the points for each lidar are loaded
    for (const auto& [lidar_id, info] : lidar_info_map) {
        EXPECT_FALSE(info.points_.empty());
    }

    // Print out the number of points for each lidar
    for (const auto& [lidar_id, info] : lidar_info_map) {
        std::cout << "Lidar ID: " << lidar_id << ", Number of points: " << info.points_.size() << std::endl;
    }

}