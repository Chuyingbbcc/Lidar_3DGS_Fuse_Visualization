//
// Created by chuchu on 8/5/26.
//

// src/main.cpp

#include <iostream>
#include <string>

#include "BundleAdjustment.h"

int main()
{
    // config.yaml's paths (e.g. "../data/...") are relative to the working
    // directory, so this must be run from the build/ directory (same
    // convention as the tests).
    const std::string config_path = "../src/config.yaml";

    BundleAdjustment bundle_adjustment(config_path);
    Status status = bundle_adjustment.Run();

    if (!status.success) {
        std::cerr << "Bundle adjustment failed: " << status.message << std::endl;
        return 1;
    }

    std::cout << "Bundle adjustment finished: " << status.message << std::endl;
    return 0;
}