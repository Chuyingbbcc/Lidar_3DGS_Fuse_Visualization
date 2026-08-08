#include "DataLoader.h"
#include <iostream>
#include <gtest/gtest.h>

TEST(DataLoaderTest, LoadLidar) {
    Config config("../src/config.yaml");
    DataLoader data_loader(config);
    auto& lidar_info_map = data_loader.get_lidar_info_map();

    // Check if the lidar_info_map_ is not empty
    EXPECT_FALSE(lidar_info_map.empty());
}