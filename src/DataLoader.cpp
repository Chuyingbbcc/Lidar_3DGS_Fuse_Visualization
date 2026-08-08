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

std::map<int, LidarPointCloudInfo>& DataLoader::get_lidar_info_map(){
 return lidar_info_map_;
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
     const fs::path depth_dir = config_.projective_z_buffer_dir_;

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
  std::string camera_img_dir = config_.input_img_dir_;
  std::string camera_timestamp_path = config_.camera_timestamp_path_;
  std::ifstream file(camera_timestamp_path);
  if (!file.is_open()) {
        throw std::runtime_error(
            "Failed to open camera timestamp file: " +
            config_.camera_timestamp_path_);
    }
    camera_map_.clear();

    std::string image_name;
    double timestamp;
    int camera_id = 0;

    while (file >> image_name >> timestamp) {

        Camera camera;
        camera.camera_id_ = camera_id;
        camera.camera_name_ = image_name;
        camera.time_stamp_ = timestamp;

       // Full image path
        camera.camera_path_ = camera_img_dir + "/" + image_name;
        // Initial pose will be filled later
        camera.initial_T_wc_ = SE3d();
        camera_map_[camera_id] = camera;
        ++camera_id;
    }
    file.close();
    std::cout << "Loaded " << camera_map_.size() << " camera images\n";
  return;
}

void DataLoader::load_lidar(){
    // load keyframes from json file
    load_keyframe_jsonl();

    // load the lidar point for each key frames [temporary disable for performance/memeroy concern]
    // for(auto& [lidar_id, info] : lidar_info_map_){
    //     lidar_info_map_[lidar_id].points_.clear();
    //     read_ply_xyz(lidar_info_map_[lidar_id].lidar_path_, lidar_info_map_[lidar_id].points_);
    // }
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

void DataLoader::read_ply_xyz(const std::string& filename , std::vector<Vec3d>& points){

    std::ifstream file(filename);

    if(!file.is_open()){
        throw std::runtime_error("Cannot open ply file: " + filename);
    }

    std::string line;
    size_t vertex_count = 0;
    while(std::getline(file, line)){
        std::stringstream ss(line);
        std::string key;
        ss >> key;
        if(key == "element"){
            std::string type;
            ss >> type;

            if(type == "vertex"){
                ss >> vertex_count;
            }
        }
        if(line == "end_header"){
            break;
        }
    }

    points.clear();
    points.reserve(vertex_count);
    for(size_t i=0; i<vertex_count; i++){
        double x,y,z;
        file >> x >> y >> z;
        points.emplace_back(x,y,z);
    }
    file.close();
}

void DataLoader::load_keyframe_jsonl(){
    // Load the keyframe JSONL file specified in the configuration
    std::ifstream file(config_.lidar_kf_path_);
    if(!file.is_open()){
        throw std::runtime_error("Cannot open keyframe JSONL file: " + config_.lidar_kf_path_);
    }
    std::string line;
    std::string json_buffer;
    int brace_count = 0;
    while(std::getline(file, line)){
        if (line.empty())
        continue;
        // Count opening/closing braces
        for (char c : line) {
            if (c == '{')
                ++brace_count;
            else if (c == '}')
                --brace_count;
        }
        json_buffer += line;
        json_buffer += '\n';
        if (brace_count == 0 && !json_buffer.empty()) {
        // Process the accumulated JSON buffer as a JSON object
            nlohmann::json json_line = nlohmann::json::parse(json_buffer);
            // Handle the JSON object as needed
            LidarPointCloudInfo info;

            info.lidar_id_ = json_line["key_frame_id"].get<int>();
            info.time_stamp_ = json_line["timestamp"].get<double>();
            info.lidar_path_ = config_.lidar_ply_dirs_+ json_line["saved_frame_path"].get<std::string>().erase(0,1);
            std::cout<<info.lidar_path_<<"\n";
            // get the LIO se3d
            auto t = json_line["lio_pose"]["translation"];
            Eigen::Vector3d translation(
                t[0].get<double>(),
                t[1].get<double>(),
                t[2].get<double>());
            auto q = json_line["lio_pose"]["quaternion_xyzw"];
            Eigen::Quaterniond rotation(
                q[3].get<double>(), // w
                q[0].get<double>(), // x
                q[1].get<double>(), // y
                q[2].get<double>()  // z
            );
            info.initial_T_wl_ = SE3d(rotation, translation);
            lidar_info_map_[info.lidar_id_] = info;
            json_buffer.clear();
        }
    }

    file.close();
    std::cout<< "Finished loading keyframe JSONL. Total keyframes: " << lidar_info_map_.size() << std::endl;
}

void DataLoader::camera_lidar_association(){
    //Build (timestamp, lidar_id) table
    std::vector<std::pair<double, int>> lidar_time_table;
    lidar_time_table.reserve(lidar_info_map_.size());

    for (const auto& [lidar_id, lidar] : lidar_info_map_){
        lidar_time_table.emplace_back(lidar.time_stamp_, lidar_id);
    }
    // sort the lidar_time_table by timestamp
    std::sort(lidar_time_table.begin(),lidar_time_table.end(),[](const auto& a, const auto& b){
            return a.first < b.first;
        });

    // associate each camera with the closest lidar timestamp
    for (auto& [camera_id, camera] : camera_map_){
        double t = camera.time_stamp_;
        auto it = std::lower_bound(lidar_time_table.begin(),lidar_time_table.end(),t,
            [](const auto& lhs, double value){
                return lhs.first < value;
            });
            
        // if the lower bound is the first element, just use it
        if(it == lidar_time_table.begin()){
            // nothing to do here, handled below
            const auto& lidar = lidar_info_map_.at(it->second);
            camera.matched_lidar_id_ = it->second;
            camera.initial_T_wc_ = extrinsic_.T_cl * lidar.initial_T_wl_.inverse();
            continue;
        }
        // if the lower bound is the end, use the last element
       if (it == lidar_time_table.end()){
            auto last = std::prev(it);
            const auto& lidar = lidar_info_map_.at(last->second);
            camera.matched_lidar_id_ = last->second;
            camera.initial_T_wc_ = extrinsic_.T_cl * lidar.initial_T_wl_.inverse();
            continue;
        }

        // Interpolate between two LiDAR poses
        auto next = it;
        auto prev = std::prev(it);

        const auto& lidar0 = lidar_info_map_.at(prev->second);
        const auto& lidar1 =lidar_info_map_.at(next->second);
        camera.matched_lidar_id_ = prev->second;

        double t0 = prev->first;
        double t1 = next->first;

        double alpha =(t - t0) / (t1 - t0);
        alpha = std::clamp(alpha, 0.0, 1.0);

        // Translation interpolation
        Eigen::Vector3d translation = (1.0 - alpha) * lidar0.initial_T_wl_.translation() +
            alpha * lidar1.initial_T_wl_.translation();

        // Rotation interpolation (SLERP)
        Eigen::Quaterniond q0 = lidar0.initial_T_wl_.unit_quaternion();
        Eigen::Quaterniond q1 = lidar1.initial_T_wl_.unit_quaternion();
        Eigen::Quaterniond q =q0.slerp(alpha, q1);

        // Interpolated LiDAR pose
        SE3d T_wl_interp(q, translation);
        // Convert LiDAR pose -> Camera pose
        camera.initial_T_wc_ = extrinsic_.T_cl * T_wl_interp.inverse();
    }
}
