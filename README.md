# LiDAR + 3D Gaussian Splatting Digital Twin

A metric-scale 3D environment for scene exploration, semantic navigation, and robot viewpoint simulation. The project combines LiDAR-initialized visual reconstruction, 3D Gaussian Splatting (3DGS), and a C++17 application built with Qt and OpenGL.

Explore a reconstructed room, inspect semantic objects, plan a route to a named target, and replay the route from the robot's viewpoint. Switch between the Gaussian scene and the colored LiDAR map while keeping the camera pose. The digital twin is intended as a foundation for perception testing, operation simulation, and synthetic data generation; the current app demonstrates scene inspection and route playback.

[Demo](#demo) · [Features](#key-features) · [Architecture](#system-architecture) · [Build](#build) · [Performance](#evaluation)

## Demo

### Search for a semantic target

Enter an object name or description to select a navigation destination, such as `go to the white mug`.

<img src="demo_video/converted_gifs/search_bar_1.gif" alt="Semantic navigation request entered in the Qt application" width="360">

### Plan a route through the scene

Define exploration paths in the reconstructed world and inspect them from above. The green control points show the editable route; the minimap provides scene context.

<img src="docs/images/navigation-path.png" alt="Top-down Gaussian scene with an editable green navigation path" width="360">

### Follow the robot's viewpoint

Play, pause, resume, or stop an exploration route. Camera position and heading update along the path, with the robot pose shown on the minimap.

<img src="demo_video/converted_gifs/navigation_35s_to_43s.gif" alt="Camera viewpoint changing during route playback" width="360">

### Navigate to an object

Select a confirmed semantic target and follow the planned route toward it. This example approaches a microwave.

<img src="demo_video/converted_gifs/navigate_to_micro_58s_to_1m09s.gif" alt="Navigation toward a microwave in the reconstructed scene" width="360">

### Inspect semantic objects

Object names, IDs, and approximate projected bounding boxes move with the view. Click a box to select an object and inspect its description.

<img src="docs/images/semantic-object-map.png" alt="Semantic labels and approximate bounding boxes over the reconstructed room" width="360">

### Compare with colored LiDAR

The colored point cloud exposes the measured scene structure. Switching representations preserves the viewpoint for visual comparison.

<img src="docs/images/colored-lidar.png" alt="Colored LiDAR scene with route and robot pose on the minimap" width="360">

[View or download the exploration video](demo_video/explore_demo.mp4).

## Key features

### Metric visual reconstruction

- **LiDAR pose priors:** initialize camera poses from a LiDAR/LIO trajectory and calibrated camera–LiDAR extrinsics to retain metric scale.
- **Visual refinement:** use multi-view feature tracks and bundle adjustment to refine camera poses and sparse landmarks.
- **Landmark management:** triangulate verified tracks, reject inconsistent observations, recover weak frames, and extend established tracks.
- **Reconstruction export:** write TUM trajectories, sparse PLY maps, COLMAP camera/image/point files, and evaluation metrics.
- **LiDAR colorization:** preparation tools project camera observations onto LiDAR geometry to produce RGB point clouds.

### LiDAR-supported 3DGS preparation

- Initialize a downstream Gaussian scene from a registered RGB LiDAR cloud using the [LiDAR dataset preparation tool](tools/lidar_init_3dgs_dataset.py).
- Generate metric depth maps, validity masks, and confidence maps for downstream depth supervision using the depth preparation scripts.
- Load the trained Gaussian PLY in the viewer. Gaussian training runs outside the C++ application; its loss configuration and training environment must be supplied separately.

### Semantic scene understanding

The project workflow uses Qwen-VL-derived object descriptions and labels to associate semantic observations with locations in the reconstructed world. The app consumes a prepared semantic database containing object classes, descriptions, confidence, review status, and approximate 3D bounds.

- Display labels and projected boxes in the Gaussian and colored LiDAR views.
- Review objects as **Confirmed**, **Unsure**, or **Incorrect**.
- Select objects directly in the viewport without moving the camera.
- Plan routes to confirmed objects using class names, aliases, description keywords, and supported spatial relations.

Semantic extraction and 3D association are upstream preparation steps; the viewer uses the resulting database for inspection and navigation.

### Qt / OpenGL navigation app

- Custom Gaussian rendering with anisotropic splats, spherical-harmonic color, opacity, depth sorting, and alpha blending.
- Perspective exploration and top-down orthographic path editing.
- Cached switching between 3DGS and colored LiDAR.
- Editable floor-aligned walkable cells and A* route planning with obstacle and clearance constraints.
- Distance-based route playback with look-ahead heading and smoothed rotation.
- A minimap showing the route, robot position, heading, walkable regions, and destination.

## System architecture

<img src="docs/system_architecture.jpg" alt="System architecture connecting visual reconstruction, 3DGS preparation, semantic mapping, and the Qt/OpenGL viewer" width="900">

*Camera observations and LiDAR pose priors feed reconstruction. Prepared Gaussian scenes, colored LiDAR, and semantic objects feed the exploration and navigation application.*

The two C++ entry points are `lio_visual_ba_pipeline` for reconstruction and `3dgs_qt_viewer` for visualization. See the [architecture reference](ARCHITECTURE.md) for data contracts and component responsibilities.

## Technical details

### Detection, matching, and tracking

```text
image -> grayscale SIFT -> grid supplementation -> descriptors
     -> BFMatcher ratio test -> pose/RANSAC geometry -> verified pairs
     -> DSU unions with camera-conflict rejection -> feature tracks
```

- **Detection:** SIFT is used as the primary detector for robust and distinctive keypoints. The image is divided into a spatial grid, and regions with insufficient SIFT coverage are supplemented with Shi–Tomasi corner features. SIFT descriptors are then computed for all keypoints, maintaining a consistent representation for downstream matching. This approach improves feature coverage and reduces clustering in highly textured regions, providing more evenly distributed geometric constraints for matching, pose estimation, and bundle adjustment. The supplemental Shi–Tomasi features can also support future KLT optical-flow tracking, enabling efficient frame-to-frame tracking for visual odometry and SLAM.

- **Matching:** Matching starts with SIFT descriptor rows from two different
  cameras. OpenCV `BFMatcher(NORM_L2)` retrieves the two closest candidates for
  each descriptor. Lowe's ratio test keeps a match only when its best distance
  is sufficiently separated from the second-best distance, removing ambiguous
  texture. Optional mutual matching performs the same search in the reverse
  direction and keeps only consistent correspondences. The result is a set of
  descriptor candidates, not yet a geometric match set.

  When calibrated camera poses are available, each candidate is tested against
  the fundamental matrix implied by the camera intrinsics and relative pose.
  The pixel Sampson error measures how far the correspondence is from its
  epipolar line; candidates beyond the configured gate are rejected. This path
  uses the LIO or optimized pose as a geometric prior and is efficient for
  temporal or recovery matching.

  For pairs without a trusted pose prior, the candidate pixels are passed to
  OpenCV `USAC_MAGSAC`. RANSAC hypotheses estimate a fundamental matrix while
  rejecting outliers, and the surviving inliers are re-evaluated with Sampson
  error. A pair is accepted only when it has enough inliers, a sufficient
  inlier ratio, and inliers distributed across both image planes. The spatial
  coverage test prevents a small, repeated texture patch from passing geometry
  despite producing a numerically valid model. Accepted correspondences carry
  their feature IDs, descriptor distance, and geometric error into pair-cache
  storage and track construction.

- **Pair selection/cache:** Temporal and pose-based loop candidates are created
  before matching. Feature and pair results use fingerprinted, checksummed
  binary caches; changed image data, feature data, or matching configuration
  invalidates reuse.

- **Tracking:** Each feature becomes a node keyed by
  `(camera_id, feature_id)`; each verified match is a graph edge. A disjoint-set
  union builds connected components while storing the camera IDs in each
  component:

  - disjoint camera sets: merge the components;
  - same root: ignore the redundant edge;
  - overlapping camera sets: reject the edge as a conflict.

  The conflict rule guarantees at most one feature observation per camera in a
  track, so later triangulation has an unambiguous correspondence. The current
  implementation is greedy: the first compatible edge in match order wins and
  a later conflicting edge is rejected. After processing, short components are
  discarded, observations and tracks receive stable camera/feature ordering,
  and a track is marked supplemental if any observation came from the grid
  supplementation stage. Accepted, redundant, and rejected edge counts are
  retained for diagnostics.

### Mapping and optimization

The post-tracking pipeline has four controlled stages:

```text
tracks -> triangulation/filtering -> sparse landmarks
    -> local/global BA -> cleanup/recovery -> final reconstruction
```

#### 1. Landmark construction

Each conflict-free track is converted into one world-space landmark. The
multi-view triangulator solves a linear DLT system from all camera observations,
then applies depth, parallax, and reprojection checks independently in every
observing camera:

```text
track observations -> DLT point X -> depth/cheirality
          -> parallax -> reprojection -> accept/reject
```

For a camera-to-world pose $T_{wc}=(R_i,C_i)$ and world point $X_j$, the projected
camera point is

$$
X_{ij}=R_i^{T}(X_j-C_i),
\qquad
\hat u_{ij}=\begin{bmatrix}
f_x X_{ij,x}/X_{ij,z}+c_x\\
f_y X_{ij,y}/X_{ij,z}+c_y
\end{bmatrix}.
$$

The observation residual is $r_{ij}=\hat u_{ij}-u_{ij}$. A candidate is
rejected if DLT is numerically unstable, any depth is non-positive, parallax is
insufficient, or the reprojection residual exceeds the configured gate. When
observation pruning is enabled, the point is fitted once, high-residual
observations are removed, and DLT is run again. The mapper records short-track,
geometric, numerical, and pruned-observation counts separately.

The geometric tests use the following quantities. For an image observation
$x_i=[u_i,v_i,1]^T$ and a camera projection matrix $P_i$, DLT constructs two
linear equations per view:

$$
\begin{bmatrix}
u_iP_{i,3}-P_{i,1}\\
v_iP_{i,3}-P_{i,2}
\end{bmatrix}X=0,
$$

where the right singular vector associated with the smallest singular value is
the homogeneous landmark estimate $X$. The singular-value check rejects
ill-conditioned configurations and points whose homogeneous scale is close to
zero. For two unit viewing rays $a$ and $b$, their parallax angle is

$$
\theta=\cos^{-1}\!\left(\operatorname{clamp}(a^Tb,-1,1)\right).
$$

The maximum angle across observing-camera pairs is retained as the track parallax.
Positive camera depth enforces cheirality. The pixel quality measures are

$$
e_{ij}=\lVert\hat u_{ij}-u_{ij}\rVert_2,
\qquad
\operatorname{RMSE}=\sqrt{\frac{1}{N}\sum_{ij}e_{ij}^{2}},
$$

with median and maximum errors retained for landmark quality and cleanup.

The same epipolar model used during matching can be written as

$$
\ell_2=Fx_1,\qquad \ell_1=F^Tx_2,
$$

where $F=K_2^{-T}EK_1^{-1}$ and $E=[t]_{\times}R$. The implemented pixel
Sampson error is the first-order approximation

$$
e_{\mathrm{S}}(x_1,x_2)=
\frac{|x_2^TFx_1|}
{\sqrt{\ell_{2,1}^{2}+\ell_{2,2}^{2}+\ell_{1,1}^{2}+\ell_{1,2}^{2}}}.
$$

This is why a correspondence can pass descriptor matching but still be removed
before tracking: its appearance similarity is acceptable, while its geometric
error is not.

#### 2. Bundle-adjustment objective

Bundle adjustment refines camera poses and landmark positions simultaneously.
The Ceres problem minimizes a robust reprojection objective with LIO pose
priors:

$$
\min_{R_i,C_i,X_j}
\sum_{(i,j)}\rho_r\left(\|r_{ij}\|^2\right)
+\sum_i\rho_p\left(\|r_i^{\mathrm{LIO}}\|^2\right)
+\sum_{j\in\mathcal P}\rho_X\left(\|(X_j-X_j^0)/\sigma_j\|^2\right).
$$

The first term connects each landmark to its observed pixels. The second keeps
optimized camera motion near the metric LIO trajectory. The third stabilizes
landmarks around their triangulated positions when an initial position exists.
$\mathcal P$ contains landmarks with an initial position, and $\sigma_j$ scales each landmark prior. Camera intrinsics remain fixed.
Camera rotations use a quaternion manifold; anchor and constant cameras are
held fixed. Reprojection uses a Cauchy loss, while pose and landmark priors use
Huber losses. Invalid-depth observations are skipped, and the candidate result
is copied back only when Ceres reports a usable solution.

For a pose prior $(q_i^0,C_i^0)$, with unit quaternions $q_i^0$ and $q_i$ representing the prior and current camera-to-world rotations, the implemented prior residual is scaled as

$$
r_i^{\mathrm{LIO}}=
\begin{bmatrix}
(C_i-C_i^0)/\sigma_t\\
2\,\operatorname{vec}\!\left((q_i^0)^{-1}q_i\right)/\sigma_r
\end{bmatrix},
$$

using the vector part of the relative quaternion for the local rotation error.
The robust loss limits the influence of mismatched observations:

$$
\rho_{\mathrm{Cauchy}}(s)=c^2\log\!\left(1+\frac{s}{c^2}\right),
\qquad
\rho_{\mathrm{Huber}}(s)=
\begin{cases}s,&s\leq c^2\\2c\sqrt{s}-c^2,&s>c^2.\end{cases}
$$

Here $s$ is a squared residual norm. Reprojection uses the Cauchy loss;
trajectory and landmark priors use Huber losses.

#### 3. Local versus global BA

The two BA modes use the same residuals and solver, but optimize different
parameter sets:

- **Local BA:** the newest camera is the seed. A temporal window ending at that
  camera is selected, and landmarks observed by the window are included. The
  window cameras and their landmarks are variable; cameras outside the window
  remain fixed but continue contributing boundary observations. This limits
  computation and lets newly added cameras adapt without moving the whole map.

- **Global BA:** the currently registered camera prefix and all eligible
  landmarks are optimized together. The anchor, constant cameras, and LIO pose
  priors keep the reconstruction metrically constrained while global landmark
  and camera drift is corrected.

- **Incremental schedule:** cameras are processed in temporal order. Local BA
  runs after a camera is registered, periodic global BA runs at configured
  checkpoints, and a final global BA runs after the last camera when needed.
  Each solve reports optimized camera/landmark counts, residual count,
  iterations, and initial/final reprojection RMSE.

#### 4. Cleanup and recovery

After each optimization phase, observations above the reprojection threshold
are removed. Landmarks that no longer have enough observations are deleted.
Weak-camera recovery then uses landmark-to-image correspondences with robust
PnP; only PnP inliers are added, followed by another BA and cleanup pass.

Optional landmark extension projects existing points into weak frames and
checks image bounds, descriptor distance, ratio consistency, and epipolar
error. Target features cannot be assigned to multiple landmarks. Accepted
observations are retriangulated and retained only if the final reprojection
test succeeds.

### Export and validation

Export writes trajectories, sparse PLY, tracks, observations, metrics, and
strict COLMAP text files for downstream reconstruction consumers. Validation independently
reopens generated artifacts and checks counts, references, finite geometry,
quaternion normalization, reprojection thresholds, and required files.

### LiDAR initialization and depth supervision

A registered RGB LiDAR cloud can replace sparse visual landmarks as the initial point set for downstream 3DGS. Preparation filters points far from the camera trajectory and isolated outliers. Separate utilities project LiDAR geometry into camera views and package metric depth, masks, and confidence maps. A compatible trainer can use these artifacts to constrain geometry during optimization; the exact depth loss and weight depend on that trainer.

### Gaussian rendering

Each Gaussian stores a position, anisotropic scale, rotation, opacity, and spherical-harmonic coefficients. Its covariance is projected into screen space:

$$
\Sigma_{3D}=RSS^TR^T,\qquad
\Sigma_{2D}=JW\Sigma_{3D}W^TJ^T.
$$

$R$ is the Gaussian rotation, $S$ its diagonal scale matrix, $W$ the rotational part of the world-to-camera transform, and $J$ the perspective-projection Jacobian. The projected covariance defines an elliptical footprint:

$$
G(x)=\exp\!\left(-\tfrac12(x-\mu)^T\Sigma_{2D}^{-1}(x-\mu)\right).
$$

The renderer draws instanced quads, evaluates view-dependent SH color, and alpha-composites depth-sorted splats. Gaussian attributes remain GPU-resident; parallel CPU depth evaluation and radix sorting update an index buffer as the view changes. GPU-assisted reordering is experimental and disabled by default.

### Coordinate systems and alignment

The LiDAR trajectory supplies the metric world frame. With $T_{AB}$ mapping coordinates from frame B to frame A, camera initialization uses:

$$
T_{WC}=T_{WL}T_{LC}.
$$

Bundle adjustment refines camera poses in this frame. Downstream geometry must use the same frame and calibration for meaningful comparison.

For navigation, the app estimates a scene-dependent vertical direction from principal components and rotates it toward +Z. A floor estimate from aligned scene geometry supports navigation height. Scene alignment and display normalization are applied consistently to objects, routes, the robot, and the minimap; saved paths are converted back to the original `3dgs_world` frame. Current Gaussian and LiDAR maps may still be only approximately registered.

### Route planning and playback

A* operates on an 8-connected, obstacle-filtered walkable grid. The planner selects reachable stopping cells near a confirmed semantic target, prevents diagonal corner cutting, and applies clearance constraints.

Playback interpolates by traveled distance, estimates heading using a look-ahead point, and smooths rotation. The simulated pose drives the camera and minimap. Pausing allows free inspection; resuming restores the route viewpoint.

## Dependencies

The current app benchmark and build were run on **Ubuntu 20.04.6 LTS**, with a C++17 compiler, CMake, Qt 5, and Mesa OpenGL. The viewer requests OpenGL 3.3 by default.

| Component | Dependencies |
|---|---|
| C++ reconstruction | Eigen3, OpenCV with SIFT, Ceres Solver, nlohmann_json, yaml-cpp, Threads |
| Qt viewer | Reconstruction dependencies, Qt 5.12+ Core/Gui/Widgets/OpenGL/Concurrent, TBB |
| Tests | GoogleTest |
| Optional preparation and inspection | Python; NumPy, SciPy, OpenCV, Matplotlib, or Pillow as required by the selected script |
| External training and semantic extraction | A compatible 3DGS trainer and the Qwen-VL preparation environment; setup is separate from the C++ build |

## Build

From the repository root, install the C++ dependencies on Ubuntu/Debian:

```bash
sudo apt update
sudo apt install build-essential cmake libeigen3-dev libopencv-dev \
  libceres-dev nlohmann-json3-dev libyaml-cpp-dev libgtest-dev \
  qtbase5-dev libqt5opengl5-dev libtbb-dev
```

Configure and build the viewer and reconstruction executable:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON -DLIO_VISUAL_BA_ENABLE_QT_VIEWER=ON
cmake --build build --target 3dgs_qt_viewer lio_visual_ba_pipeline -j2
```

To build and run the tests:

```bash
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

## Evaluation

### App performance

| Scene | Gaussians | GPU | Average FPS | Mean frame time (ms) | P95 frame time (ms) | Load time (s) | Peak RAM (MiB) |
|---|---:|---|---:|---:|---:|---:|---:|
| 3DGS demo, continuous rotation | 1,814,073 | Intel UHD 620 | 7.58 | 131.88 | 158.62 | 4.356 | 1,952.8 |

**Test configuration:** Intel Core i7-8650U · Intel UHD Graphics 620 · 1100 × 700 requested window · 1,814,073 Gaussians · LiDAR points: not included in this benchmark.

Results are means across three Release-build runs per mode on Ubuntu 20.04.6 / Mesa 21.2.6. Each rendering run discards 10 warm-up frames and measures 120 rotating frames using CPU splat reordering. P95 is the mean of the three per-run P95 values. Loading measures demo setup through the first swapped frame in a fresh process, without clearing the OS file cache. RAM is peak process resident memory; application GPU memory is not reported separately on this shared-memory GPU.

LiDAR rendering, warm view-switch latency, and A* planning latency were not measured in this evaluation.

### KITTI OXTS evaluation — Drive 0035

The Drive 0035 results compare the LIO-prior and optimized camera trajectories
against the KITTI OXTS reference after rigid alignment. OXTS is an independent
reference, not survey-grade camera ground truth. The optimized trajectory
reduces the median and P90 trajectory, orientation, local-motion, and jitter errors. Maximum ATE increases from 0.1677 m to 0.1878 m.

**Trajectory error against the OXTS reference**

ATE denotes absolute trajectory error; P90 and P95 are the 90th and 95th percentiles.

| Metric | LIO prior | Optimized pose (round 2) |
|---|---:|---:|
| ATE median (m) | 0.0492 | 0.0403 |
| ATE P90 (m) | 0.0942 | 0.0789 |
| ATE max (m) | 0.1677 | 0.1878 |
| Orientation error median (deg) | 1.3061 | 0.6043 |
| Orientation error P90 (deg) | 1.5857 | 0.9126 |
| One-frame translation error median (m) | 0.0230 | 0.0206 |
| One-frame translation error P90 (m) | 0.0419 | 0.0404 |
| One-frame rotation error median (deg) | 0.0971 | 0.0147 |
| One-frame rotation error P90 (deg) | 0.1591 | 0.0778 |
| Position jitter median (m) | 0.0166 | 0.0079 |
| Position jitter P90 (m) | 0.0451 | 0.0211 |

**Reconstruction quality across optimization rounds**

| Metric | First round | Second round / final |
|---|---:|---:|
| Landmarks | 19,678 | 19,647 |
| Observations | 152,278 | 168,326 |
| Reprojection median (px) | 0.3938 | 0.2834 |
| Reprojection P90 (px) | 1.0913 | 0.8282 |
| Reprojection P95 (px) | — | 1.0943 |
| Reprojection max (px) | — | 2.0000 |
| Validation | pass | pass |

**Pose correction relative to the LIO prior**

| Quantity | Samples | RMSE | Median | P90 | P95 | Max |
|---|---:|---:|---:|---:|---:|---:|
| Translation displacement | 120 | 0.0328 m | 0.0262 m | 0.0460 m | 0.0568 m | 0.0823 m |
| Rotation displacement | 120 | 0.4875 deg | 0.4476 deg | 0.6620 deg | 0.7428 deg | 0.8363 deg |

The first table measures trajectory error against OXTS, the second measures
reconstruction quality across optimization rounds, and the third measures how
far optimization moved the LIO-prior poses. Displacement is a correction
magnitude, not an absolute accuracy measure.

### Evaluation scope

The KITTI tables describe Drive 0035, a separate dataset from the indoor scene used for the app benchmark. Missing first-round P95 and maximum reprojection errors are shown as em dashes.

Dataset image counts, LiDAR input size, landmark distributions, BA runtime/memory results, and training resource measurements for the featured indoor scene are not yet reported here. The app benchmark measures visualization performance; it does not establish reconstruction accuracy.

## Limitations

- **Geometric degeneracy:** planar walls, limited structure, and low-texture regions can leave pose and geometry estimates weakly constrained.
- **Time synchronization:** camera–LiDAR timestamp errors during motion degrade pose consistency and image-to-map alignment.
- **Extrinsic calibration:** camera–LiDAR rotation or translation errors can produce misalignment, ghosting, and blurred reconstruction.
- **Feature tracking:** motion blur, lighting changes, low texture, and changing image resolution can break visual correspondences. Object-level and semantic tracking are future directions.
- **Photometric consistency:** exposure, white balance, and lighting differences across views can cause inconsistent colors and reconstruction artifacts.
- **Coverage and overlap:** unseen regions and limited viewpoint diversity can leave holes, floaters, or missing geometry.
- **Camera pose accuracy:** small pose errors can blur surfaces, duplicate structures, and reduce detail in the trained Gaussian scene.
- **Rendering hardware:** the measured integrated-GPU result is about 7.6 FPS; smoother rendering at higher resolutions needs more graphics capacity or a smaller workload.
- **Training resources:** 3DGS optimization can require substantial GPU memory. The trainer, training configuration, and measured VRAM usage are not supplied here.
- **Navigation scope:** walkable cells and Gaussian-derived clearance are experimental scene-planning aids. Playback simulates a camera/robot viewpoint; physical robot control, collision certification, and a complete dynamics simulator are outside the current app.
