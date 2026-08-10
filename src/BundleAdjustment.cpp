//
// Created by chuchu on 8/5/26.
//
#include "BundleAdjustment.h"
#include "General.h"
#include  "BaHelper.h"
#include "DataLoader.h"
#include "Optimizer.h"
#include "SIFT.h"
#include "DataType.h"
#include <iostream>
#include <map>
#include <algorithm>
#include <cmath>

using namespace std;
BundleAdjustment::BundleAdjustment(std::string config_path): config_(config_path), data_loader_(config_) {

}

Status BundleAdjustment::Run(){
   Status status;
   status = data_loader_.load();
   if(!status.success){
      cout<<status.message<<endl;
      return status;
   }
   //
   auto& camera_map = data_loader_.get_camera_map();
   CameraIntrinsic cam_intrinsic = data_loader_.get_intrinsic();

   // Snapshot the pre-optimization trajectory so it can be compared
   // against the optimized result later.
   status = BaHelper::save_trajectory(camera_map, "../data/trajectory/initial.txt", false);
   if(!status.success){
      cout << status.message << endl;
   }
   //Sift
   for(auto& it : camera_map){
       Camera& camera=  it.second;
       status = SIFT::extract_sift(camera, config_);
       if(!status.success){
         cout<<status.message<<endl;
          return status;
       }
   }

   //featrue matching - resume from cache if a previous run already saved one
   std::map<int, Landmark> landmarks;
   status = SIFT::load_landmarks(config_.landmark_cache_path_, landmarks);
   if(!status.success){
      cout << "[BA] no usable landmark cache, running exhaustive matching: " << status.message << endl;
      status = SIFT::exhaust_pair_matching_parallel(camera_map, config_, landmarks);
      if(!status.success){
         cout<<status.message<<endl;
            return status;
      }
      status = SIFT::save_landmarks(config_.landmark_cache_path_, landmarks);
      if(!status.success){
         cout << status.message << endl;
      }
   } else {
      cout << status.message << endl;
   }
   status = BaHelper::update_observation_depth(camera_map, landmarks, true);
   //initial the world pos
   status = BaHelper::extract_landmark_world_pos(camera_map, cam_intrinsic, config_, landmarks, true);
   if(!status.success){
         cout<<status.message<<endl;
          return status;
    }

   //optimize
   optimizer_.SetCameraMap(camera_map);
   optimizer_.SetLandmarkMap(landmarks);
   optimizer_.SetMaxIterations(config_.ceres_max_iterations_);
   optimizer_.SetIntrinsicMatrix(cam_intrinsic.K);

   bool optimizer_initialized = false;
   double prev_cost = 0.0;

   for(int i=0; i<config_.num_iteration_; i++){

      // Optimize()'s "initialized" flag means "bootstrap from initial_*
      // fields"; optimizer_initialized instead tracks "optimized_* fields
      // have already been populated by a previous call", so pass negated.
      status= optimizer_.Optimize(!optimizer_initialized);
      if(!status.success){
            cout<<status.message<<endl;
         return status;
      }

      const double curr_cost = optimizer_.GetFinalCost();
      cout << "[BA] iteration " << i << " final cost: " << curr_cost << endl;

      // update the camera/ landmark map
      optimizer_.GetCameraMap(camera_map);
      optimizer_.GetLandmarkMap(landmarks);

      // Record this iteration's camera trajectory for later visualization.
      const std::string trajectory_path = "../data/trajectory/iter_" + std::to_string(i) + ".txt";
      status = BaHelper::save_trajectory(camera_map, trajectory_path, true);
      if(!status.success){
         cout << status.message << endl;
      }

      // set new optimized camera map to dataloader for depth update
      data_loader_.set_camera_map(camera_map);
      optimizer_initialized = true;
      data_loader_.update_depth_map_parallel(optimizer_initialized);
      // after new depth map generated, need to update the landmark map(obervation depth info todo)
      status = BaHelper::update_observation_depth(camera_map, landmarks, false);
      status = BaHelper::extract_landmark_world_pos(camera_map, cam_intrinsic, config_, landmarks,false);
      // set new updated camera/landmark map back to optimizer
      optimizer_.SetCameraMap(camera_map);
      optimizer_.SetLandmarkMap(landmarks);

      // Stop early once the cost stops improving meaningfully.
      // if(i > 0){
      //    const double relative_change = std::abs(prev_cost - curr_cost) / std::max(prev_cost, 1e-12);
      //    if(relative_change < config_.ba_cost_threshold_){
      //       cout << "[BA] converged early at iteration " << i
      //            << " (relative cost change " << relative_change << ")" << endl;
      //       prev_cost = curr_cost;
      //       break;
      //    }
      // }
      prev_cost = curr_cost;
   }

   //get finalized optimzied camera pose
   std::map<int, SE3d>optimzied_poses;
   optimizer_.GetOptimizedPoses(optimzied_poses);
   status = BaHelper::writeOptimziedCamera(camera_map, config_);
   if(!status.success){
      cout << status.message << endl;
   }

   // Snapshot the final optimized trajectory for easy comparison against
   // ../data/trajectory/initial.txt.
   status = BaHelper::save_trajectory(camera_map, "../data/trajectory/final.txt", true);
   if(!status.success){
      cout << status.message << endl;
   }

   return status;
}


