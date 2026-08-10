

#include <map>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <ceres/ceres.h>

#include "DataType.h"
#include "Optimizer.h"


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
void BundleAdjustmentOptimizer::SetMaxIterations(int max_iterations){
    max_num_iterations_ = max_iterations;
    return;
}
double BundleAdjustmentOptimizer::GetFinalCost() const{
    return last_final_cost_;
}

    // Run bundle adjustment
Status BundleAdjustmentOptimizer::Optimize(const bool initialized){
    if(camera_map_.empty() || landmark_map_.empty()){
        return {false, "camera_map_ or landmark_map_ is empty"};
    }

    // camera_poses_ / landmark_positions_ are keyed by camera_id / landmark_id
    camera_poses_.clear();
    for(const auto& [camera_id, camera] : camera_map_){
        if(!initialized){
            camera_poses_[camera_id] = camera.optimized_T_cw_.log();
        }else{
            camera_poses_[camera_id] = camera.initial_T_cw_.log();
        }
    }

    landmark_positions_.clear();
    for(const auto& [landmark_id, landmark] : landmark_map_){
        if(!initialized){
            landmark_positions_[landmark_id] = landmark.optimized_position_;
        }else{
            landmark_positions_[landmark_id] = landmark.initial_position_;
        }
    }

    ceres::Problem problem;

    // Track residual blocks by cost type so we can report which one
    // dominates the total cost, since they're otherwise summed together.
    std::vector<ceres::ResidualBlockId> reprojection_blocks;
    std::vector<ceres::ResidualBlockId> depth_blocks;
    std::vector<ceres::ResidualBlockId> pose_prior_blocks;

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
            reprojection_blocks.push_back(problem.AddResidualBlock(
                cost_function,
                new ceres::HuberLoss(1.0),
                camera_poses_[obs.camera_id_].data(),
                landmark_positions_[landmark_id].data()));

            double curr_dpeth = obs.depth_;
            if(!initialized){
                curr_dpeth = obs.optimized_depth_;
            }
            if(curr_dpeth > 0.0){
                ceres::CostFunction* depth_cost_function =
                    new ceres::AutoDiffCostFunction<DepthError, 1, 6, 3>(
                        new DepthError(curr_dpeth, depth_prior_weight_));

                depth_blocks.push_back(problem.AddResidualBlock(
                    depth_cost_function,
                    new ceres::HuberLoss(1.0),
                    camera_poses_[obs.camera_id_].data(),
                    landmark_positions_[landmark_id].data()));
            }
        }
    }

    // Anchor each camera to its initial (prior) pose. Reprojection error alone is only
    // defined up to a global SE3 transform *and* scale (classic monocular SFM gauge
    // ambiguity); the pose prior keeps every pose near its prior and resolves it.
    for(auto& [camera_id, camera] : camera_map_){
        auto curr_camera_pose = camera.initial_T_cw_;
        if(!initialized){
            curr_camera_pose = camera.optimized_T_cw_;
        }
        ceres::CostFunction* prior_cost_function =
            new ceres::AutoDiffCostFunction<PosePriorError, 6, 6>(
                new PosePriorError(curr_camera_pose, pose_prior_weight_));

        pose_prior_blocks.push_back(problem.AddResidualBlock(
            prior_cost_function,
            nullptr,
            camera_poses_[camera_id].data()));
    }

    // Sums 0.5 * sum(residual^2) over just this group of blocks, matching
    // ceres::Solver::Summary's cost convention.
    auto evaluate_group_cost = [&problem](const std::vector<ceres::ResidualBlockId>& blocks) -> double {
        if(blocks.empty()){
            return 0.0;
        }
        ceres::Problem::EvaluateOptions eval_options;
        eval_options.residual_blocks = blocks;
        double cost = 0.0;
        problem.Evaluate(eval_options, &cost, nullptr, nullptr, nullptr);
        return cost;
    };

    const double initial_reprojection_cost = evaluate_group_cost(reprojection_blocks);
    const double initial_depth_cost = evaluate_group_cost(depth_blocks);
    const double initial_pose_prior_cost = evaluate_group_cost(pose_prior_blocks);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.max_num_iterations = max_num_iterations_;
    // Print Ceres's own per-iteration cost report (cost, cost change, trust
    // region radius, etc.) to stdout as it solves.
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    last_final_cost_ = summary.final_cost;

    // Ceres already tracks initial/final cost for this single Solve() call,
    // so report the change here instead of relying on the caller to diff
    // GetFinalCost() across outer-loop iterations.
    const double solve_relative_change =
        std::abs(summary.initial_cost - summary.final_cost) /
        std::max(summary.initial_cost, 1e-12);

    std::cout
        << "[Optimizer] initial cost: " << summary.initial_cost
        << " -> final cost: " << summary.final_cost
        << " (relative change: " << solve_relative_change
        << ", iterations: " << summary.num_successful_steps + summary.num_unsuccessful_steps
        << ")" << std::endl;

    const double final_reprojection_cost = evaluate_group_cost(reprojection_blocks);
    const double final_depth_cost = evaluate_group_cost(depth_blocks);
    const double final_pose_prior_cost = evaluate_group_cost(pose_prior_blocks);

    std::cout
        << "[Optimizer] cost breakdown (reprojection / depth prior / pose prior): "
        << initial_reprojection_cost << " -> " << final_reprojection_cost << "  |  "
        << initial_depth_cost << " -> " << final_depth_cost << "  |  "
        << initial_pose_prior_cost << " -> " << final_pose_prior_cost
        << std::endl;

    // write optimized results back into camera_map_ / landmark_map_
    for(const auto& [camera_id, xi] : camera_poses_){
        camera_map_[camera_id].optimized_T_cw_ = SE3d::exp(xi);
        camera_map_[camera_id].optimized_ = true;
    }
    for(const auto& [landmark_id, pos] : landmark_positions_){
        landmark_map_[landmark_id].optimized_position_ = pos;
        landmark_map_[landmark_id].optimized_ = true;
    }

    // Report the actual pixel-space reprojection error (project each
    // optimized landmark into each observing camera and compare against the
    // observed pixel). This is the direct geometric error, distinct from the
    // weighted/Huberized Ceres cost above.
    // {
    //     double sum_sq_error = 0.0;
    //     double max_error = 0.0;
    //     int count = 0;
    //     int behind_camera = 0;

    //     for(const auto& [landmark_id, landmark] : landmark_map_){
    //         for(const auto& obs : landmark.observations_){
    //             auto camera_it = camera_map_.find(obs.camera_id_);
    //             if(camera_it == camera_map_.end()){
    //                 continue;
    //             }

    //             const Vec3d point_c =
    //                 camera_it->second.optimized_T_cw_ * landmark.optimized_position_;

    //             if(point_c.z() <= 0.0){
    //                 ++behind_camera;
    //                 continue;
    //             }

    //             const double u = K_(0, 0) * point_c.x() / point_c.z() + K_(0, 2);
    //             const double v = K_(1, 1) * point_c.y() / point_c.z() + K_(1, 2);
    //             const double du = u - obs.pixel_.x();
    //             const double dv = v - obs.pixel_.y();
    //             const double error = std::sqrt(du * du + dv * dv);
    //             // std::cout<<landmark.optimized_position_ << std::endl;
    //             // std::cout << "point_c: " << point_c.transpose() << std::endl;
    //             // std::cout<< K_ << std::endl;
    //             // std::cout << "[Optimizer] reprojection error for landmark " << landmark_id
    //             //           << " in camera " << obs.camera_id_ << ": " <<"u: " << u << ", v: " << v << ", error: " << error << ", observed u: " << obs.pixel_.x() << ", observed v : " << obs.pixel_.y() << std::endl;

    //             sum_sq_error += error * error;
    //             max_error = std::max(max_error, error);
    //             ++count;
    //         }
    //     }

    //     if(count > 0){
    //         const double rmse = std::sqrt(sum_sq_error / count);
    //         std::cout
    //             << "[Optimizer] pixel reprojection error - RMSE: " << rmse
    //             << " px, max: " << max_error << " px, over " << count
    //             << " observations";
    //         if(behind_camera > 0){
    //             std::cout << " (" << behind_camera << " observations skipped: behind camera)";
    //         }
    //         std::cout << std::endl;
    //     }
    // }

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

void BundleAdjustmentOptimizer::GetCameraMap(std::map<int, Camera>& camera_map) const{
    camera_map = camera_map_;
    return;
}

void BundleAdjustmentOptimizer::GetLandmarkMap(std::map<int, Landmark>& landmark_map) const{
    landmark_map = landmark_map_;
    return;
}