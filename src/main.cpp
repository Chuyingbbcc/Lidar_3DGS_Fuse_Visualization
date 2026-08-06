//
// Created by chuchu on 8/5/26.
//

// src/main.cpp

#include <iostream>

#include <opencv2/core.hpp>

#include <Eigen/Core>

#include <ceres/ceres.h>

#include "BundleAdjustment.h"

int main()
{
    std::cout << "========== Library Check ==========\n";

    // OpenCV
    std::cout << "OpenCV version : " << CV_VERSION << '\n';

    // Eigen
    Eigen::Vector3d v(1.0, 2.0, 3.0);
    std::cout << "Eigen vector   : "
              << v.transpose() << '\n';

    // Ceres
    std::cout << "Ceres version  : "
              << CERES_VERSION_STRING << '\n';

    std::cout << "\nAll libraries linked successfully!\n";

    return 0;
}