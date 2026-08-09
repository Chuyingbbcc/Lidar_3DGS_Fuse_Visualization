#include "Optimizer.h"

#include <map>

#include <ceres/ceres.h>

#include "DataType.h"

struct BundleAdjustmentOptimizer::ReprojectionError{
    ReprojectionError(const Vec2d& pixel,const Mat3d& K): pixel_(pixel),K_(K){}

    template<typename T>
    bool operator()(const T* const camera,const T* const landmark,
                    T* residuals) const{
        //---------------------------------------
        // camera as Sophus SE3 tangent vector (xi)
        //---------------------------------------
        // camera[0:5] : se3 tangent (omega, upsilon)
        ConstVec6Map<T> xi(camera);
        SE3<T> T_cw = SE3<T>::exp(xi);

        // landmark[0:2] : world point being optimized
        ConstVec3Map<T> point_w(landmark);
        Vec3<T> point_c = T_cw * point_w;

        //---------------------------------------
        // projection
        //---------------------------------------

        T x = point_c[0] / point_c[2];
        T y = point_c[1] / point_c[2];

        T u = T(K_(0,0)) * x + T(K_(0,2));
        T v = T(K_(1,1)) * y + T(K_(1,2));

        residuals[0] = u - T(pixel_.x());
        residuals[1] = v - T(pixel_.y());

        return true;
    }

private:

    Vec2d pixel_;

    Mat3d K_;
};

struct BundleAdjustmentOptimizer::PosePriorError{
    PosePriorError(const SE3d& prior, double weight): prior_(prior),weight_(weight){}

    template<typename T>
    bool operator()(const T* const pose, T* residuals) const {

        //----------------------------------
        // Convert optimized xi to SE3
        //----------------------------------

        ConstVec6Map<T> xi(pose);

        SE3<T> current_pose = SE3<T>::exp(xi);

        //----------------------------------
        // Convert prior SE3d -> SE3<T>
        //----------------------------------

        SE3<T> prior_pose = prior_.template cast<T>();

        //----------------------------------
        // SE3 error
        //----------------------------------

        SE3<T> error = prior_pose.inverse() * current_pose;

        Vec6<T> error_vec = error.log();

        //----------------------------------
        // residual
        //----------------------------------

        Vec6Map<T> residual(residuals);

        residual = T(weight_) * error_vec;

        return true;
    }

private:
    SE3d prior_;
    double weight_;
};

struct BundleAdjustmentOptimizer::DepthError{
    DepthError(double depth, double weight): depth_(depth), weight_(weight){}

    template<typename T>
    bool operator()(const T* const camera, const T* const landmark, T* residuals) const {
        ConstVec6Map<T> xi(camera);
        SE3<T> T_cw = SE3<T>::exp(xi);

        ConstVec3Map<T> point_w(landmark);
        Vec3<T> point_c = T_cw * point_w;

        // Anchors landmark depth to the LiDAR depth-map measurement, removing the
        // scale ambiguity that reprojection error alone leaves unconstrained.
        residuals[0] = T(weight_) * (point_c.z() - T(depth_));

        return true;
    }

private:
    double depth_;
    double weight_;
};

void BundleAdjustmentOptimizer::SetCameraMap(const std::map<int, Camera>& camera_map){
    camera_map_ = camera_map;
    return;
}

void BundleAdjustmentOptimizer::SetLandmarkMap(const std::map<int, Landmark>& landmark_map){
    landmark_map_ = landmark_map;
    return;
}
void BundleAdjustmentOptimizer::SetIntrinsicMatrix(const Mat3d& K){
    K_ = K;
    return;
}
void BundleAdjustmentOptimizer::SetPosePriorWeight(double weight){
    pose_prior_weight_ = weight;
    return;
}
void BundleAdjustmentOptimizer::SetDepthPriorWeight(double weight){
    depth_prior_weight_ = weight;
    return;
}

    // Run bundle adjustment
Status BundleAdjustmentOptimizer::Optimize(){
    if(camera_map_.empty() || landmark_map_.empty()){
        return {false, "camera_map_ or landmark_map_ is empty"};
    }

    // camera_poses_ / landmark_positions_ are keyed by camera_id / landmark_id
    camera_poses_.clear();
    for(const auto& [camera_id, camera] : camera_map_){
        camera_poses_[camera_id] = camera.initial_T_cw_.log();
    }

    landmark_positions_.clear();
    for(const auto& [landmark_id, landmark] : landmark_map_){
        landmark_positions_[landmark_id] = landmark.initial_position_;
    }

    ceres::Problem problem;

    for(auto& [landmark_id, landmark] : landmark_map_){
        for(const Observation& obs : landmark.observations_){
            auto camera_it = camera_map_.find(obs.camera_id_);
            if(camera_it == camera_map_.end()){
                continue;
            }

            ceres::CostFunction* cost_function =
                new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6, 3>(
                    new ReprojectionError(obs.pixel_, K_));

            // Huber loss guards against outlier feature matches dominating the solution
            problem.AddResidualBlock(
                cost_function,
                new ceres::HuberLoss(1.0),
                camera_poses_[obs.camera_id_].data(),
                landmark_positions_[landmark_id].data());

            if(obs.depth_ > 0.0){
                ceres::CostFunction* depth_cost_function =
                    new ceres::AutoDiffCostFunction<DepthError, 1, 6, 3>(
                        new DepthError(obs.depth_, depth_prior_weight_));

                problem.AddResidualBlock(
                    depth_cost_function,
                    new ceres::HuberLoss(1.0),
                    camera_poses_[obs.camera_id_].data(),
                    landmark_positions_[landmark_id].data());
            }
        }
    }

    // Anchor each camera to its initial (prior) pose. Reprojection error alone is only
    // defined up to a global SE3 transform *and* scale (classic monocular SFM gauge
    // ambiguity); the pose prior keeps every pose near its prior and resolves it.
    for(auto& [camera_id, camera] : camera_map_){
        ceres::CostFunction* prior_cost_function =
            new ceres::AutoDiffCostFunction<PosePriorError, 6, 6>(
                new PosePriorError(camera.initial_T_cw_, pose_prior_weight_));

        problem.AddResidualBlock(
            prior_cost_function,
            nullptr,
            camera_poses_[camera_id].data());
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // write optimized results back into camera_map_ / landmark_map_
    for(const auto& [camera_id, xi] : camera_poses_){
        camera_map_[camera_id].optimized_T_cw_ = SE3d::exp(xi);
        camera_map_[camera_id].optimized_ = true;
    }
    for(const auto& [landmark_id, pos] : landmark_positions_){
        landmark_map_[landmark_id].optimized_position_ = pos;
        landmark_map_[landmark_id].optimized_ = true;
    }

    return {true, summary.BriefReport()};
}

    // Optional helper if caller only wants optimized poses
void BundleAdjustmentOptimizer::GetOptimizedPoses(
        std::map<int, SE3d>& optimized_poses) const{
    optimized_poses.clear();
    for(const auto& [camera_id, camera] : camera_map_){
        optimized_poses[camera_id] = camera.optimized_T_cw_;
    }
    return;
}

    // Optional helper if caller only wants optimized landmark positions
void BundleAdjustmentOptimizer::GetOptimizedLandmarks(
        std::map<int, Vec3d>& optimized_landmarks) const{
    optimized_landmarks.clear();
    for(const auto& [landmark_id, landmark] : landmark_map_){
        optimized_landmarks[landmark_id] = landmark.optimized_position_;
    }
    return;
}