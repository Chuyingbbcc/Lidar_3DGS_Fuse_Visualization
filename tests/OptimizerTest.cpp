//
// Synthetic bundle-adjustment data test for BundleAdjustmentOptimizer.
//
#include <gtest/gtest.h>
#include "Optimizer.h"

namespace {

Mat3d MakeIntrinsics() {
    Mat3d K = Mat3d::Identity();
    K(0, 0) = 500.0;  // fx
    K(1, 1) = 500.0;  // fy
    K(0, 2) = 320.0;  // cx
    K(1, 2) = 240.0;  // cy
    return K;
}

// Projects a world point into pixel coordinates given a world-to-camera pose.
Vec2d Project(const SE3d& T_cw, const Vec3d& point_w, const Mat3d& K) {
    Vec3d point_c = T_cw * point_w;
    double x = point_c.x() / point_c.z();
    double y = point_c.y() / point_c.z();
    return Vec2d(K(0, 0) * x + K(0, 2), K(1, 1) * y + K(1, 2));
}

}  // namespace

TEST(BundleAdjustmentOptimizerTest, RecoversPosesAndLandmarksFromSyntheticData) {
    const Mat3d K = MakeIntrinsics();

    // Ground-truth world-to-camera poses.
    std::map<int, SE3d> gt_poses;
    gt_poses[0] = SE3d(SO3d::exp(Vec3d(0.0, 0.0, 0.0)), Vec3d(0.0, 0.0, 0.0));
    gt_poses[1] = SE3d(SO3d::exp(Vec3d(0.05, -0.03, 0.02)), Vec3d(0.5, 0.1, -0.2));
    gt_poses[2] = SE3d(SO3d::exp(Vec3d(-0.04, 0.06, 0.01)), Vec3d(-0.3, 0.2, 0.3));

    // Ground-truth landmarks (in front of every camera).
    std::map<int, Vec3d> gt_landmarks;
    gt_landmarks[0] = Vec3d(0.2, 0.1, 5.0);
    gt_landmarks[1] = Vec3d(-0.3, 0.4, 6.0);
    gt_landmarks[2] = Vec3d(0.5, -0.2, 4.5);
    gt_landmarks[3] = Vec3d(-0.1, -0.3, 5.5);
    gt_landmarks[4] = Vec3d(0.0, 0.0, 7.0);

    // Distinct per-camera perturbations seed the optimizer away from ground truth.
    // A shared perturbation across all cameras would be an unobservable global SE3
    // gauge transform (reprojection error is invariant to it), which the pose prior
    // could never be pulled away from; per-camera perturbations break that symmetry
    // so reprojection data can correct each pose individually.
    std::map<int, SE3d> pose_perturbations;
    pose_perturbations[0] = SE3d::exp(Vec6d(0.01, -0.01, 0.02, 0.03, -0.02, 0.01));
    pose_perturbations[1] = SE3d::exp(Vec6d(-0.02, 0.015, -0.01, -0.02, 0.03, -0.015));
    pose_perturbations[2] = SE3d::exp(Vec6d(0.015, 0.02, -0.02, 0.025, -0.01, 0.02));

    std::map<int, Camera> camera_map;
    for (const auto& [camera_id, T_cw] : gt_poses) {
        Camera camera;
        camera.camera_id_ = camera_id;
        camera.camera_name_ = "cam" + std::to_string(camera_id);
        camera.initial_T_wc_ = pose_perturbations.at(camera_id) * T_cw;
        camera_map[camera_id] = camera;
    }

    std::map<int, Landmark> landmark_map;
    for (const auto& [landmark_id, point_w] : gt_landmarks) {
        Landmark landmark;
        landmark.landmark_id = landmark_id;

        // Rough initial guess (no triangulation/depth modeled here) to seed the solver.
        landmark.initial_position = point_w + Vec3d(0.1, -0.1, 0.2);

        for (const auto& [camera_id, T_cw] : gt_poses) {
            Observation obs;
            obs.camera_id = camera_id;
            obs.keypoint_idx = landmark_id;
            obs.pixel = Project(T_cw, point_w, K);
            // Simulates a LiDAR depth-map sample at this pixel: a real, independent
            // metric measurement, not derived from the (perturbed) camera pose.
            obs.depth = (T_cw * point_w).z();
            landmark.observations.push_back(obs);
        }

        landmark_map[landmark_id] = landmark;
    }

    BundleAdjustmentOptimizer optimizer;
    optimizer.SetIntrinsicMatrix(K);
    optimizer.SetCameraMap(camera_map);
    optimizer.SetLandmarkMap(landmark_map);
    optimizer.SetPosePriorWeight(10.0);
    optimizer.SetDepthPriorWeight(10.0);

    Status status = optimizer.Optimize();
    ASSERT_TRUE(status.success) << status.message;

    std::map<int, SE3d> optimized_poses;
    optimizer.GetOptimizedPoses(optimized_poses);

    // Reprojection error alone is invariant to a rigid transform of the whole scene
    // (cameras + landmarks together), so PosePriorError - a soft constraint - can
    // only pull the result to within the residual bias of the (noisy) priors, not
    // to an exact match with ground truth. The depth residual pins landmark scale,
    // so landmarks converge much tighter than the pose gauge bias allows for poses.
    constexpr double kPoseTolerance = 0.05;
    constexpr double kLandmarkTolerance = 0.1;

    for (const auto& [camera_id, T_gt] : gt_poses) {
        const SE3d& T_opt = optimized_poses.at(camera_id);
        SE3d pose_error = T_gt.inverse() * T_opt;
        EXPECT_LT(pose_error.log().norm(), kPoseTolerance)
            << "camera " << camera_id << " pose did not converge";
    }

    std::map<int, Vec3d> optimized_landmarks;
    optimizer.GetOptimizedLandmarks(optimized_landmarks);

    for (const auto& [landmark_id, point_gt] : gt_landmarks) {
        const Vec3d& point_opt = optimized_landmarks.at(landmark_id);
        EXPECT_LT((point_gt - point_opt).norm(), kLandmarkTolerance)
            << "landmark " << landmark_id << " did not converge";
    }
}
