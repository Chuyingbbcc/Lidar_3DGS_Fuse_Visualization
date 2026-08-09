//
// Created by chuchu on 8/5/26.
//

#include "BaHelper.h"
#include <iostream>
using namespace std;

namespace {
// Weiszfeld's algorithm: iteratively reweighted average that converges to the
// point minimizing sum of distances, making it robust to outlier points
// (unlike the arithmetic mean, which minimizes sum of squared distances).
Vec3d geometric_median(const std::vector<Vec3d>& points, int max_iters = 50, double eps = 1e-6){
   if(points.size() == 1){
      return points.front();
   }

   Vec3d median = Vec3d::Zero();
   for(const auto& p : points) median += p;
   median /= static_cast<double>(points.size());

   for(int iter = 0; iter < max_iters; ++iter){
      Vec3d numerator = Vec3d::Zero();
      double denominator = 0.0;
      for(const auto& p : points){
         double dist = (p - median).norm();
         if(dist < 1e-9) continue; // avoid divide-by-zero when a point coincides with the current estimate
         numerator += p / dist;
         denominator += 1.0 / dist;
      }
      if(denominator < 1e-9) break;

      Vec3d next = numerator / denominator;
      bool converged = (next - median).norm() < eps;
      median = next;
      if(converged) break;
   }
   return median;
}
}

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
   for(auto& [lm_id, lm] : landmarks){

      if(lm.observations_.size() < 2){
          continue;
      }

      std::vector<Vec3d> world_points;
      world_points.reserve(lm.observations_.size());

      for(auto& ob : lm.observations_){
         //for each pixel, camera<-pixel, world<-camera
         double depth = ob.depth_;
         if(!is_initial){
            depth = ob.optimized_depth_;
         }

         Vec3d cam_cor = PixelToCamera(ob.pixel_,depth,intrinsic.K);

         //for orb, get camera cvt to world coordinate
         if(camera_map.find(ob.camera_id_)==camera_map.end()){
            std::cout<<"obj:" << ob.camera_id_<<" "<< "can not find camera: " << ob.camera_id_ << std::endl;
            continue;
         }
         const Camera& cam = camera_map.at(ob.camera_id_);
         const SE3d& T_cw = is_initial ? cam.initial_T_cw_ : cam.optimized_T_cw_;
         SE3d T_wc = T_cw.inverse();
         //comvert it to world coordinate
         world_points.push_back(T_wc * cam_cor);
      }

      if(world_points.empty()){
         continue;
      }

      // Geometric median instead of mean: robust to outlier observations
      // (e.g. LiDAR depth noise near edges) since it minimizes sum of
      // distances rather than squared distances.
      Vec3d pos_w_medium = geometric_median(world_points);

      if(is_initial){
       lm.initial_position_ = pos_w_medium;
      }else{
       lm.optimized_ = true;
       lm.optimized_position_ = pos_w_medium;
      }
   }// loop end, land mark
   return {true, "OK"};
}


Status  BaHelper::writeOptimziedCamera(std::map<int ,Camera>& camera_map, Config& config){
  return {true, "write done"};
}

Status BaHelper::update_observation_depth(const std::map<int ,Camera>& camera_map, std::map<int, Landmark>& landmarks, bool is_initial){
   for(auto& [lm_id, lm] : landmarks){
      for(auto& ob : lm.observations_){
         if(camera_map.find(ob.camera_id_) == camera_map.end()){
            std::cout<<"obj:" << ob.camera_id_<<" "<< "can not find camera: " << ob.camera_id_ << std::endl;
            continue;
         }
         const Camera& cam = camera_map.at(ob.camera_id_);
         if(cam.depth_map_.empty()){
            continue;
         }

         const int u = static_cast<int>(std::round(ob.pixel_.x()));
         const int v = static_cast<int>(std::round(ob.pixel_.y()));
         if(u < 0 || u >= cam.depth_map_.cols || v < 0 || v >= cam.depth_map_.rows){
            continue;
         }

         const float depth = cam.depth_map_.at<float>(v, u);
         if(depth <= 0.0f){
            continue;
         }

         if(is_initial){
            ob.depth_ = static_cast<double>(depth);
         }else{
            ob.optimized_depth_ = static_cast<double>(depth);
            ob.optimized_ = true;
         }
      }
   }
   return {true, "observation depth updated"};
}








