//
// Created by chuchu on 8/5/26.
//

#include "BaHelper.h"
#include <iostream>
using namespace std;
Status BaHelper::load_projected_depth(Camera& camera){
return {true, "depth load ok!"};
}

Vec3d BaHelper::PixelToCamera(const Vec2d& pixel,double depth,const Mat3d& K){
  double fx = K(0,0);
  double fy = K(1,1);

  double cx = K(0,2);
  double cy = K(1,2);

  double x =
        (pixel.x() - cx) / fx;

  double y =
        (pixel.y() - cy) / fy;

  return Vec3d(
        x * depth,
        y * depth,
        depth);
}

Status BaHelper::extract_landmark_world_pos(const std::map<int ,Camera>& camera_map, const CameraIntrinsic& intrinsic, const Config& config_, std::map<int, Landmark>&landmarks,  bool is_initial){
   //get camera intrinsic
   for(auto it: landmarks){
      int lm_id = it.first;
      Landmark& lm = it.second;

      Vec3d pos_w_medium = Vec3d::Zero();
      if(lm.observations_.size() < 2){
          continue;
      }
      for(auto& ob : lm.observations_){
         //for each pixel, camera<-pixel, world<-camera
         double depth = ob.depth_;

         Vec3d cam_cor = PixelToCamera(ob.pixel_,depth,intrinsic.K);

         //for orb, get camera cvt to world coordinate
         if(camera_map.find(ob.camera_id_)==camera_map.end()){
            std::cout<<"obj:" << ob.camera_id_<<" "<< "can not find camera: " << ob.camera_id_ << std::endl;
            continue;
         }
         const Camera& cam = camera_map.at(ob.camera_id_);
         SE3d T_cam_w = SE3d();
         //SE3d T_cam_w  = is_initial? cam.initial_T_cam_w_ : cam.T_cam_w_;
         SE3d T_w_cam = T_cam_w.inverse();
         //comvert it to world coordinate
         Vec3d w_cor = T_w_cam * cam_cor;
         pos_w_medium += w_cor;
      }
      pos_w_medium /= static_cast<double>(lm.observations_.size());

      if(is_initial){
       lm.initial_position_ = pos_w_medium;
      }
      else{
       lm.optimized_ = true;
       lm.optimized_position_ = pos_w_medium;
      }
   }// loop end, land mark
}


Status  BaHelper::writeOptimziedCamera(std::map<int ,Camera>& camera_map, Config& config){
  return {true, "write done"};
}








