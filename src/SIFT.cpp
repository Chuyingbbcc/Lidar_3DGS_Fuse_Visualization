//
// Created by chuchu on 8/5/26.
//


#include "General.h"
#include "SIFT.h"

using namespace std;
Status SIFT::extract_sift( Camera& camera){
   //Todo:: add implementation
   return {true, "SIFT OK!"};
}

Status SIFT::exhaust_pair_matching(
    const std::map<int, Camera>& camera_map,
    Config& config,std::map<int, Landmark>& landmarks){
    //Todo:: add implementation
    return {true, "Match OK!"};
}
