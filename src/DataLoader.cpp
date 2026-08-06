//
// Created by chuchu on 8/5/26.
//

#include "DataLoader.h"
#include <string>
#include <fstream>
#include <stdexcept>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include "General.h"
#include <filesystem>


namespace fs = std::filesystem;
 Status DataLoader::load(){
  //fill the camera map
  intrinsic_ = load_intrinsic();
  extrinsic_ = load_extrinsic();
  load_camera();
  load_depth();
  return {true, "OK"};
}


std::map<int, Camera>& DataLoader::get_camera_map(){
 return camera_map_;
}

CameraIntrinsic DataLoader::load_intrinsic(){
  std::string intrinsic_path  = config_.camera_intrinsic_path_;
     std::ifstream fin(intrinsic_path);
     if (!fin.is_open()) {
         throw std::runtime_error(
             "Cannot open intrinsic file: " + intrinsic_path);
     }

     std::string line;
     CameraIntrinsic camera_intrinsic;
     for (int row = 0; row < 3; ++row) {

         if (!std::getline(fin, line)) {
             throw std::runtime_error(
                 "Invalid intrinsic file.");
         }

         // Replace ',' with ' '
         for (char& c : line) {
             if (c == ',')
                 c = ' ';
         }

         std::stringstream ss(line);

         for (int col = 0; col < 3; ++col) {
             if (!(ss >> camera_intrinsic.K(row, col))) {
                 throw std::runtime_error(
                     "Failed to parse intrinsic matrix.");
             }
         }
     }

     std::cout << "Camera intrinsic loaded:\n"
               << camera_intrinsic.K << std::endl;
  return camera_intrinsic;
}

CameraExtrinsic DataLoader::load_extrinsic(){
  std::string extrinsic_path = config_.camera_extrinsic_path_;
     std::ifstream input_file(extrinsic_path);

     if (!input_file.is_open()) {
         throw std::runtime_error(
             "Failed to open extrinsic file: " + extrinsic_path
         );
     }

     nlohmann::json root;

     try {
         input_file >> root;
     } catch (const nlohmann::json::parse_error& error) {
         throw std::runtime_error(
             "Failed to parse extrinsic JSON: " +
             std::string(error.what())
         );
     }

     if (!root.contains("T_camera_lidar")) {
         throw std::runtime_error(
             "Missing T_camera_lidar in: " + extrinsic_path
         );
     }

     Mat4d T_CL_matrix =
         load_matrix4d(root.at("T_camera_lidar"));

     const Mat3d R_CL =
         T_CL_matrix.block<3, 3>(0, 0);

     const Vec3d t_CL =
         T_CL_matrix.block<3, 1>(0, 3);

     CameraExtrinsic extrinsic;

     extrinsic.T_camera_lidar =
         SE3d(R_CL, t_CL);

     // Compute the inverse instead of trusting a second stored matrix.
     extrinsic.T_lidar_camera =
         extrinsic.T_camera_lidar.inverse();

     return extrinsic;
}

void DataLoader::load_depth(){
     int loaded_count = 0;
     int missing_count = 0;
  const fs::path depth_dir =
       config_.projective_z_buffer_dir_;

     if (!fs::exists(depth_dir)) {
         throw std::runtime_error(
             "Depth directory does not exist: " +
             depth_dir.string()
         );
     }

     if (!fs::is_directory(depth_dir)) {
         throw std::runtime_error(
             "Depth path is not a directory: " +
             depth_dir.string()
         );
     }
    for (auto& [camera_id, camera] : camera_map_) {
        const fs::path image_path(camera.camera_path_);

        // Original:
        // /image_dir/IMG_2129.jpg
        //
        // Depth:
        // /depth_dir/IMG_2129.png
        const fs::path depth_path =
            depth_dir /
            (image_path.stem().string() + ".png");

        if (!fs::exists(depth_path)) {
            std::cerr
                << "[load_depth] Missing depth image: "
                << depth_path << '\n';

            camera.depth_map_.release();
            ++missing_count;
            continue;
        }

        cv::Mat depth_map = cv::imread(
            depth_path.string(),
            cv::IMREAD_UNCHANGED
        );

        if (depth_map.empty()) {
            std::cerr
                << "[load_depth] Failed to read: "
                << depth_path << '\n';

            camera.depth_map_.release();
            ++missing_count;
            continue;
        }

        if (depth_map.channels() != 1) {
            std::cerr
                << "[load_depth] Depth image must be single-channel: "
                << depth_path
                << ", channels = "
                << depth_map.channels()
                << '\n';

            camera.depth_map_.release();
            ++missing_count;
            continue;
        }

        // Optional: verify depth resolution matches original image.
        cv::Mat image = cv::imread(
            camera.camera_path_,
            cv::IMREAD_COLOR
        );

        if (!image.empty() &&
            image.size() != depth_map.size()) {
            std::cerr
                << "[load_depth] Size mismatch for camera "
                << camera_id
                << ". Image: "
                << image.cols << " x " << image.rows
                << ", depth: "
                << depth_map.cols << " x " << depth_map.rows
                << '\n';

            camera.depth_map_.release();
            ++missing_count;
            continue;
        }

        // cv::Mat uses reference-counted memory.
        // This modifies the Camera object inside camera_dict_.
        camera.depth_map_ = depth_map;

        ++loaded_count;
    }

    std::cout
        << "[load_depth] Loaded "
        << loaded_count
        << " depth maps, missing/invalid "
        << missing_count
        << std::endl;
  return;
}

void DataLoader::load_camera(){
  std::string camera_path = config_.input_img_dir_;
     //Todo:: add implementation
  return;
}

Mat4d DataLoader::load_matrix4d(
    const nlohmann::json& json_matrix){
     if (!json_matrix.is_array() || json_matrix.size() != 4) {
         throw std::runtime_error(
             "Expected a 4x4 matrix in the extrinsic JSON."
         );
     }

     Mat4d matrix;

     for (int row = 0; row < 4; ++row) {
         if (!json_matrix[row].is_array() ||
             json_matrix[row].size() != 4) {
             throw std::runtime_error(
                 "Expected a 4x4 matrix in the extrinsic JSON."
             );
             }

         for (int col = 0; col < 4; ++col) {
             matrix(row, col) =
                 json_matrix[row][col].get<double>();
         }
     }
     return matrix;
}
