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

   //Sift
   auto& camera_map = data_loader_.get_camera_map();
   for(auto& it : camera_map){
       Camera& camera=  it.second;
       status = SIFT::extract_sift(camera);
       if(!status.success){
         cout<<status.message<<endl;
          return status;
       }
       status = BaHelper::load_projected_depth(camera);
       if(!status.success){
         cout<<status.message<<endl;
          return status;
       }
   }

   //featrue matching
   std::map<int, Landmark> landmarks;
   status = SIFT::exhaust_pair_matching(camera_map, config_, landmarks);
    if(!status.success){
         cout<<status.message<<endl;
          return status;
    }

   //initial the world pos
   status = BaHelper::extract_initial_landmark_world_pos(camera_map, config_, landmarks);
   if(!status.success){
         cout<<status.message<<endl;
          return status;
    }

   //optimize
   optimizer_.SetCameraMap(camera_map);
   optimizer_.SetLandmarkMap(landmarks);
   status= optimizer_.Optimize();
   if(!status.success){
   cout<<status.message<<endl;
    return status;
   }

   //get optimzied camera pose
   std::map<int, SE3d>optimzied_poses;
   optimizer_.GetOptimizedPoses(optimzied_poses);
   //BaHelper::writeOptimziedCamera(, config_.output_path_);

}


