//
// Created by chuchu on 8/5/26.
//

#include "DataLoader.h"
#include <string>
#include "General.h"

 Status DataLoader::load(){
  //fill the camera map
  load_camera();
  load_intrinsic();
  load_extrinsic();
  load_depth();
  return {true, "OK"};
}


std::map<int, Camera>& DataLoader::get_camera_map(){
 return camera_map_;
}

void DataLoader::load_intrinsic(){
  std::string intrinsic_path  = config_.camera_intrinsic_path_;
  //Todo:: add implementation
  return;
}

void DataLoader::load_extrinsic(){
  std::string extrinsic_path = config_.camera_extrinsic_path_;
   //Todo:: add implementation
  return;
}
void DataLoader::load_depth(){
  std::string depth_dir = config_.projective_z_buffer_dir_;
   //Todo:: add implementation
  return;
}

void DataLoader::load_camera(){
  std::string camera_path = config_.input_img_dir_;
     //Todo:: add implementation
  return;
}


