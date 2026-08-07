//
// Created by chuchu on 8/5/26.
//

#include "General.h"
#include "SIFT.h"

#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>

#include <functional>
#include <iostream>
#include <iterator>
#include <map>
#include <set>
#include <utility>
#include <vector>


// ============================================================
// Extract SIFT
// ============================================================

Status SIFT::extract_sift(
    Camera& camera,
    Config& config)
{
    //----------------------------------------------------------
    // Load image
    //----------------------------------------------------------
    cv::Mat image = cv::imread(
        camera.camera_path_,
        cv::IMREAD_GRAYSCALE);

    if (image.empty())
    {
        return {
            false,
            "Failed to load image: " + camera.camera_path_
        };
    }

    //----------------------------------------------------------
    // Create SIFT detector
    //----------------------------------------------------------
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create(
        config.sift_nfeatures_,
        config.sift_n_octave_layers_,
        config.sift_contrast_threshold_,
        config.sift_edge_threshold_,
        config.sift_sigma_);

    //----------------------------------------------------------
    // Clear previous result
    //----------------------------------------------------------
    camera.keypoints_.clear();
    camera.descriptors_.release();

    //----------------------------------------------------------
    // Extract
    //----------------------------------------------------------
    sift->detectAndCompute(
        image,
        cv::noArray(),
        camera.keypoints_,
        camera.descriptors_);

    if (camera.keypoints_.empty())
    {
        return {
            false,
            "No SIFT features detected in image: "
                + camera.camera_name_
        };
    }

    if (camera.descriptors_.empty())
    {
        return {
            false,
            "Failed to compute SIFT descriptors for image: "
                + camera.camera_name_
        };
    }

    return {
        true,
        "Extracted "
            + std::to_string(camera.keypoints_.size())
            + " SIFT features from "
            + camera.camera_name_
    };
}


// ============================================================
// KNN matching
// ============================================================

void SIFT::knn_matching(
    const cv::Mat& desc_1,
    const cv::Mat& desc_2,
    std::vector<std::vector<cv::DMatch>>& knn_matches)
{
    knn_matches.clear();

    //----------------------------------------------------------
    // SIFT uses floating-point descriptors, so use L2 distance
    //----------------------------------------------------------
    cv::BFMatcher matcher(cv::NORM_L2);

    matcher.knnMatch(
        desc_1,
        desc_2,
        knn_matches,
        2);
}


// ============================================================
// Lowe ratio test
// ============================================================

void SIFT::lowe_ratio_test(
    const std::vector<std::vector<cv::DMatch>>& knn_matches,
    std::vector<cv::DMatch>& ratio_matches,
    float ratio_threshold)
{
    ratio_matches.clear();

    for (const auto& matches : knn_matches)
    {
        //------------------------------------------------------
        // Need first and second nearest neighbors
        //------------------------------------------------------
        if (matches.size() < 2)
        {
            continue;
        }

        const cv::DMatch& best = matches[0];
        const cv::DMatch& second = matches[1];

        //------------------------------------------------------
        // Lowe ratio test
        //------------------------------------------------------
        if (best.distance <
            ratio_threshold * second.distance)
        {
            ratio_matches.push_back(best);
        }
    }
}


// ============================================================
// Match -> pixel coordinates
// ============================================================

void SIFT::match_to_pixels(
    const std::vector<cv::DMatch>& ratio_matches,
    const Camera& camera_1,
    const Camera& camera_2,
    std::vector<cv::Point2f>& points_1,
    std::vector<cv::Point2f>& points_2)
{
    points_1.clear();
    points_2.clear();

    points_1.reserve(ratio_matches.size());
    points_2.reserve(ratio_matches.size());

    for (const auto& match : ratio_matches)
    {
        //------------------------------------------------------
        // queryIdx -> camera 1 keypoint
        //------------------------------------------------------
        points_1.push_back(
            camera_1.keypoints_[match.queryIdx].pt);

        //------------------------------------------------------
        // trainIdx -> camera 2 keypoint
        //------------------------------------------------------
        points_2.push_back(
            camera_2.keypoints_[match.trainIdx].pt);
    }
}


// ============================================================
// Geometric verification
// ============================================================

int SIFT::geometric_verification(
    const std::vector<cv::Point2f>& points_1,
    const std::vector<cv::Point2f>& points_2,
    double ransac_threshold,
    std::vector<uchar>& inlier_mask)
{
    inlier_mask.clear();

    //----------------------------------------------------------
    // Fundamental matrix requires at least 8 points for the
    // standard 8-point formulation
    //----------------------------------------------------------
    if (points_1.size() < 8 ||
        points_2.size() < 8)
    {
        return 0;
    }

    //----------------------------------------------------------
    // Estimate F and reject geometrically inconsistent matches
    //----------------------------------------------------------
    cv::Mat F = cv::findFundamentalMat(
        points_1,
        points_2,
        cv::FM_RANSAC,
        ransac_threshold,
        0.99,
        inlier_mask);

    if (F.empty())
    {
        return 0;
    }

    //----------------------------------------------------------
    // Count RANSAC inliers
    //----------------------------------------------------------
    int num_inliers = 0;

    for (uchar value : inlier_mask)
    {
        if (value)
        {
            ++num_inliers;
        }
    }

    return num_inliers;
}


// ============================================================
// Union matches
// ============================================================

void SIFT::union_matches(
    int camera_id_1,
    int camera_id_2,
    const std::vector<cv::DMatch>& ratio_matches,
    const std::vector<uchar>& inlier_mask,
    std::map<FeatureNode, FeatureNode>& parent)
{
    //----------------------------------------------------------
    // Rank is used only to make tree merging more balanced.
    //
    // It does not represent camera or feature information.
    //----------------------------------------------------------
    std::map<FeatureNode, int> rank;


    //----------------------------------------------------------
    // Find root
    //----------------------------------------------------------
    std::function<FeatureNode(const FeatureNode&)> find_root;

    find_root =
        [&](const FeatureNode& node) -> FeatureNode
    {
        //------------------------------------------------------
        // First time seeing this feature
        //------------------------------------------------------
        if (parent.find(node) == parent.end())
        {
            parent[node] = node;
            rank[node] = 0;
        }

        //------------------------------------------------------
        // Path compression
        //------------------------------------------------------
        if (parent[node] != node)
        {
            parent[node] = find_root(parent[node]);
        }

        return parent[node];
    };


    //----------------------------------------------------------
    // Union two feature nodes
    //----------------------------------------------------------
    auto unite =
        [&](const FeatureNode& a,
            const FeatureNode& b)
    {
        FeatureNode root_a = find_root(a);
        FeatureNode root_b = find_root(b);

        if (root_a == root_b)
        {
            return;
        }

        //------------------------------------------------------
        // Union by rank
        //------------------------------------------------------
        if (rank[root_a] < rank[root_b])
        {
            std::swap(root_a, root_b);
        }

        parent[root_b] = root_a;

        if (rank[root_a] == rank[root_b])
        {
            ++rank[root_a];
        }
    };


    //----------------------------------------------------------
    // Add only RANSAC inlier matches
    //----------------------------------------------------------
    const size_t count =
        std::min(
            ratio_matches.size(),
            inlier_mask.size());

    for (size_t i = 0; i < count; ++i)
    {
        if (!inlier_mask[i])
        {
            continue;
        }

        const cv::DMatch& match =
            ratio_matches[i];

        //------------------------------------------------------
        // A feature is identified by:
        //
        // (camera id, keypoint index)
        //------------------------------------------------------
        FeatureNode node_1{
            camera_id_1,
            match.queryIdx
        };

        FeatureNode node_2{
            camera_id_2,
            match.trainIdx
        };

        unite(
            node_1,
            node_2);
    }
}


// ============================================================
// Extract Landmark map
// ============================================================

void SIFT::extract_landmark_map(
    const std::map<FeatureNode, FeatureNode>& parent,
    const std::map<int, Camera>& camera_map,
    std::map<int, Landmark>& landmark_map)
{
    landmark_map.clear();

    //----------------------------------------------------------
    // Nothing matched
    //----------------------------------------------------------
    if (parent.empty())
    {
        return;
    }


    //----------------------------------------------------------
    // Helper for finding the final root.
    //
    // parent is const here, so we don't do path compression.
    //----------------------------------------------------------
    auto find_root =
        [&parent](FeatureNode node)
    {
        auto it = parent.find(node);

        while (it != parent.end() &&
               it->second != node)
        {
            node = it->second;
            it = parent.find(node);
        }

        return node;
    };


    //----------------------------------------------------------
    // Group features according to Union-Find root
    //----------------------------------------------------------
    std::map<
        FeatureNode,
        std::vector<FeatureNode>
    > tracks;

    for (const auto& entry : parent)
    {
        const FeatureNode& node =
            entry.first;

        FeatureNode root =
            find_root(node);

        tracks[root].push_back(node);
    }


    //----------------------------------------------------------
    // Convert tracks to Landmark objects
    //----------------------------------------------------------
    int landmark_id = 0;

    for (const auto& track_entry : tracks)
    {
        const std::vector<FeatureNode>& track =
            track_entry.second;

        //------------------------------------------------------
        // One feature alone is not a multi-view landmark
        //------------------------------------------------------
        if (track.size() < 2)
        {
            continue;
        }

        Landmark landmark;
        landmark.landmark_id_ = landmark_id;

        //------------------------------------------------------
        // One physical point should occur at most once
        // in each camera.
        //------------------------------------------------------
        std::set<int> used_cameras;

        for (const FeatureNode& node : track)
        {
            const int camera_id =
                node.first;

            const int keypoint_index =
                node.second;


            //--------------------------------------------------
            // Avoid duplicate observations from same camera
            //--------------------------------------------------
            if (used_cameras.count(camera_id) > 0)
            {
                continue;
            }


            //--------------------------------------------------
            // Make sure camera exists
            //--------------------------------------------------
            auto camera_it =
                camera_map.find(camera_id);

            if (camera_it == camera_map.end())
            {
                continue;
            }

            const Camera& camera =
                camera_it->second;


            //--------------------------------------------------
            // Make sure keypoint index is valid
            //--------------------------------------------------
            if (keypoint_index < 0 ||
                keypoint_index >=
                    static_cast<int>(
                        camera.keypoints_.size()))
            {
                continue;
            }

            used_cameras.insert(camera_id);


            //--------------------------------------------------
            // Create observation
            //--------------------------------------------------
            Observation observation;

            observation.camera_id_ =
                camera_id;

            observation.keypoint_idx_ =
                keypoint_index;

            const cv::Point2f& pt = camera.keypoints_[keypoint_index].pt;

            observation.pixel_ = Vec2d( static_cast<double>(pt.x), static_cast<double>(pt.y));

            landmark.observations_.push_back(
                observation);
        }


        //------------------------------------------------------
        // Need observations from at least two cameras
        //------------------------------------------------------
        if (landmark.observations_.size() < 2)
        {
            continue;
        }


        //------------------------------------------------------
        // Add landmark
        //------------------------------------------------------
        landmark_map.emplace(
            landmark_id,
            std::move(landmark));

        ++landmark_id;
    }
}


// ============================================================
// Exhaustive pair matching
// ============================================================

Status SIFT::exhaust_pair_matching(
    const std::map<int, Camera>& camera_map,
    Config& config,
    std::map<int, Landmark>& landmarks)
{
    //----------------------------------------------------------
    // Config
    //----------------------------------------------------------
    const float ratio_threshold =
        config.ratio_threshold_;

    const double ransac_threshold =
        config.ransac_threshold_;

    const int min_inliers =
        config.min_inliers_;


    landmarks.clear();


    //----------------------------------------------------------
    // Union-Find must live across ALL image pairs.
    //
    // Very important:
    // Do NOT create this inside the inner loop.
    //----------------------------------------------------------
    std::map<
        FeatureNode,
        FeatureNode
    > parent;


    //----------------------------------------------------------
    // Exhaustive image pair matching
    //----------------------------------------------------------
    for (auto it1 = camera_map.begin();
         it1 != camera_map.end();
         ++it1)
    {
        const int camera_id_1 =
            it1->first;

        const Camera& camera_1 =
            it1->second;


        //------------------------------------------------------
        // Skip camera without descriptors
        //------------------------------------------------------
        if (camera_1.descriptors_.empty())
        {
            continue;
        }


        //------------------------------------------------------
        // Start from next camera.
        //
        // This avoids:
        //
        // 0-0
        // 1-1
        //
        // and duplicated pairs:
        //
        // 0-1 and 1-0
        //------------------------------------------------------
        auto it2 = std::next(it1);

        for (; it2 != camera_map.end(); ++it2)
        {
            const int camera_id_2 =
                it2->first;

            const Camera& camera_2 =
                it2->second;


            if (camera_2.descriptors_.empty())
            {
                continue;
            }


            // =================================================
            // 1. KNN descriptor matching
            // =================================================

            std::vector<
                std::vector<cv::DMatch>
            > knn_matches;

            knn_matching(
                camera_1.descriptors_,
                camera_2.descriptors_,
                knn_matches);


            // =================================================
            // 2. Lowe ratio test
            // =================================================

            std::vector<cv::DMatch>
                ratio_matches;

            lowe_ratio_test(
                knn_matches,
                ratio_matches,
                ratio_threshold);


            if (static_cast<int>(
                    ratio_matches.size()) <
                min_inliers)
            {
                continue;
            }


            // =================================================
            // 3. Convert matches to pixels
            // =================================================

            std::vector<cv::Point2f>
                points_1;

            std::vector<cv::Point2f>
                points_2;

            match_to_pixels(
                ratio_matches,
                camera_1,
                camera_2,
                points_1,
                points_2);


            // =================================================
            // 4. Fundamental matrix + RANSAC
            // =================================================

            std::vector<uchar>
                inlier_mask;

            int num_inliers =
                geometric_verification(
                    points_1,
                    points_2,
                    ransac_threshold,
                    inlier_mask);


            if (num_inliers < min_inliers)
            {
                continue;
            }


            // =================================================
            // 5. Add verified matches to Union-Find
            // =================================================

            union_matches(
                camera_id_1,
                camera_id_2,
                ratio_matches,
                inlier_mask,
                parent);


            //--------------------------------------------------
            // Debug output
            //--------------------------------------------------
            std::cout
                << "Camera "
                << camera_id_1
                << " <-> "
                << camera_id_2
                << " | ratio matches: "
                << ratio_matches.size()
                << " | inliers: "
                << num_inliers
                << std::endl;
        }
    }


    //----------------------------------------------------------
    // Convert tracks to landmark map
    //----------------------------------------------------------
    extract_landmark_map(
        parent,
        camera_map,
        landmarks);


    //----------------------------------------------------------
    // Summary
    //----------------------------------------------------------
    std::cout
        << "Generated "
        << landmarks.size()
        << " landmark tracks."
        << std::endl;


    return {
        true,
        "Feature matching done! Generated "
            + std::to_string(landmarks.size())
            + " landmarks."
    };
}