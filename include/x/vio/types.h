/*
 * Copyright 2020 California  Institute  of Technology (“Caltech”)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef X_VIO_TYPES_H
#define X_VIO_TYPES_H

#include <x/vision/types.h>
#include <x/vision/feature.h>
#include <x/vision/tiled_image.h>

#include <vector>
#include <Eigen/Dense>
#include <easy/profiler.h>
#include <x/common/csv_writer.h>
#if __has_include(<filesystem>)
  #include <filesystem>
  namespace fs = std::filesystem;
  #elif __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
  namespace fs = std::experimental::filesystem;
#else
  error "Missing the <filesystem> header."
#endif

/**
 * This header defines common types used in xVIO.
 */
namespace x
{
  using FeaturesCsv = CsvWriter<profiler::timestamp_t, size_t, size_t, size_t, size_t>;
  using TracksCsv = CsvWriter<profiler::timestamp_t, u_int32_t, double, double, double, double, double, std::string>;


  struct XVioPerformanceLogger {

    explicit XVioPerformanceLogger(const fs::path & path)
     : features_csv(path / "features.csv", {"ts", "num_slam_features", "num_msckf_features", "num_opportunistic_features", "num_potential_features"})
     , tracks_csv(path / "xvio_tracks.csv", {"lost_ts", "id", "t", "x", "y", "x_dist", "y_dist", "update_type"}) {}

    FeaturesCsv features_csv;
    TracksCsv tracks_csv;
  };

  typedef std::shared_ptr<XVioPerformanceLogger> XVioPerformanceLoggerPtr;
  /**
   * Enum carrying camera distortion model to use. X default is FOV
   */
  enum class DistortionModel : char {
    FOV,
    RADIAL_TANGENTIAL,
  };

  enum class HasteTrackerType : char {
    CORRELATION,
    HASTE_CORRELATION,
    HASTE_CORRELATION_STAR,
    HASTE_DIFFERENCE,
    HASTE_DIFFERENCE_STAR,
  };

  enum class AsyncFrontendFeatureInterpolation : char {
    NO_INTERPOLATION, // always take last
    NEAREST_NEIGHBOR, // take one of both
    LINEAR_NO_LIMIT,  // does linear inter- and extrapolation
    LINEAR_RELATIVE_LIMIT, // does linear inter- and extrapolation till a ratio of interpolation_limit is not exceeded
    LINEAR_ABSOLUTE_LIMIT, // does linear inter- and extrapolation until interpolation_limit in seconds is not exceeded
  };

  enum class AsyncFrontendUpdateStrategy : char {
    EVERY_ROS_EVENT_MESSAGE,
    EVERY_N_EVENTS,
    // triggers EKF update when incoming event timestamps are more than n msec ahead from previous update
    EVERY_N_MSEC_WITH_EVENTS,
  };

  enum class AsyncFrontendUpdateTimestamp : char {
    PATCH_AVERAGE,
    PATCH_MAXIMUM,
  };

  enum class EkltPatchTimestampAssignment : char {
    LATEST_EVENT,
    ACCUMULATED_EVENTS_CENTER,
  };


  /**
   * Async frontend interpolation + update options: how HASTE and EKLT are used to generate EKF updates
   */
  struct AsyncFrontendParams {

    /**
     * Params for feature detection from APS frames
     */
    int detection_min_distance = 30; // Minimum distance between detected features. Parameter passed to goodFeatureToTrack
    int detection_harris_block_size = 30; // Block size to compute Harris score. passed to harrisCorner and goodFeaturesToTrack
    double detection_harris_k = 0; // Magic number for Harris score
    double detection_harris_quality_level = 0; // Determines range of harris score allowed between the maximum and minimum. Passed to goodFeaturesToTrack

    bool enable_outlier_removal = false; // Whether or not to remove outliers with the same method as xVIO

    // for now same x::Params.outlier_method [...]
    int outlier_method;
    double outlier_param1;
    double outlier_param2;

    AsyncFrontendFeatureInterpolation ekf_feature_interpolation = AsyncFrontendFeatureInterpolation::LINEAR_NO_LIMIT;
    // factor limiting the extrapolation amount. E.g. 0.0 means only interpolation is performed (no extrapolation), 1.0
    // means that at most the time difference between the last two points is used for extrapolation.
    // If < 0, no limit is applied.
    double ekf_feature_extrapolation_limit = -1.0;
    int ekf_update_every_n = -1;

    AsyncFrontendUpdateStrategy ekf_update_strategy = AsyncFrontendUpdateStrategy::EVERY_ROS_EVENT_MESSAGE;
    AsyncFrontendUpdateTimestamp ekf_update_timestamp = AsyncFrontendUpdateTimestamp::PATCH_AVERAGE;
  };

  /**
   * User-defined parameters
   */
  struct Params
  {
    Vector3 p;
    Vector3 v;
    Quaternion q;
    Vector3 b_w;
    Vector3 b_a;
    Vector3 sigma_dp;
    Vector3 sigma_dv;
    Vector3 sigma_dtheta;
    Vector3 sigma_dbw;
    Vector3 sigma_dba;
    double cam_fx;
    double cam_fy;
    double cam_cx;
    double cam_cy;

    /**
     * Defines which distortion model to use, parameter vector needs to be filled accordingly.
     * Default is FOV
     */
    DistortionModel cam_distortion_model = DistortionModel::FOV;

    /**
     * Contains as many parameters as the distortion model requires
     */
    std::vector<double> cam_distortion_parameters;
    int img_height;
    int img_width;
    Vector3 p_ic;
    Quaternion q_ic;
   
    /**
     * Standard deviation of feature measurement [in normalized coordinates]
     */
    double sigma_img;

    /**
     * Standard deviation of range measurement noise [m].
     */
    double sigma_range;

    Quaternion q_sc;
    Vector3 w_s;
    double n_a;
    double n_ba;
    double n_w;
    double n_bw;
    int fast_detection_delta;
    bool non_max_supp;
    int block_half_length;
    int margin;
    int n_feat_min;
    int outlier_method;
    double outlier_param1;
    double outlier_param2;
    int n_tiles_h;
    int n_tiles_w;
    int max_feat_per_tile;
    double time_offset;

    /**
     * Maximum number of poses in the sliding window.
     */
    int n_poses_max;

    /**
     * Maximum number of SLAM features.
     */
    int n_slam_features_max;

    /**
     * Initial inverse depth of SLAM features [1/m].
     *
     * This is when SLAM features can't be triangulated for MSCKK-SLAM. By
     * default, this should be 1 / (2 * d_min), with d_min the minimum
     * expected feature depth (2-sigma) [Montiel, 2006]
     */
    double rho_0;

    /**
     * Initial standard deviation of SLAM inverse depth [1/m].
     *
     * This is when SLAM features can't be triangulated for MSCKK-SLAM. By
     * default, this should be 1 / (4 * d_min), with d_min the minimum
     * expected feature depth (2-sigma) [Montiel, 2006].
     */
    double sigma_rho_0;

    /**
     * Number of IEKF iterations (EKF <=> 1).
     */
    int iekf_iter;

    /**
     * Minimum baseline to trigger MSCKF measurement (pixels).
     */
    double msckf_baseline;

    /**
     * Minimum track length for a visual feature to be processed.
     */
    int min_track_length;

    /**
     * Gravity vector in world frame [x,y,z]m in m/s^2
     */
    Vector3 g;

    /**
     * Max specific force norm threshold, after which accelerometer readings are detected as spikes. [m/s^2]
     */
    double a_m_max = 50.0;

    /**
     * State buffer size (default: 250 states)
     */
    int state_buffer_size = 250;

    /**
     * Expected difference between successive IMU sequence IDs.
     *
     * Used to detects missing IMU messages. Default value 1.
     */
    int delta_seq_imu = 1;

    /**
     * Time margin, in seconds, around the buffer time range.
     *
     * This sets the tolerance for how far in the future/past the closest
     * state request can be without returning an invalid state.
     */
    double state_buffer_time_margin = 0.02;

    double traj_timeout_gui;
    bool init_at_startup;

    /**
     * Event related parameters.
     */
    /*double event_cam_fx;
    double event_cam_fy;
    double event_cam_cx;
    double event_cam_cy;
    DistortionModel event_cam_distortion_model = DistortionModel::FOV;
    std::vector<double> event_cam_distortion_parameters;
    int event_img_height;
    int event_img_width;
    Vector3 p_ec;
    Quaternion q_ec;
    double event_cam_time_offset;*/

    int event_accumulation_method;
    double event_accumulation_period;
    int n_events_min;
    int n_events_max;
    int noise_event_rate;
    int n_events_noise_detection_min;
    double event_norm_factor;
    bool correct_event_motion;


    /**
     * EKLT frontend parameters
     */
    // feature detection
    int eklt_max_corners = 100; // Maximum features allowed to be tracked
    int eklt_min_corners = 60; // Minimum features allowed to be tracked

    // tracker
    double eklt_log_eps = 1e-2; // Small constant to compute log image. To avoid numerical issues normally we compute log(img /255 + log_eps)
    bool eklt_use_linlog_scale = false; // whether to use piecewise lin-log scale instead of log(img+log_eps)
    double eklt_first_image_t = -1; // If specified discards all images until this time.

    // tracker
    int eklt_lk_window_size = 15; // Parameter for KLT. Used for bootstrapping feature.
    int eklt_num_pyramidal_layers = 2; // Parameter for KLT. Used for bootstrapping feature.
    int eklt_batch_size = 200; // Determines the size of the event buffer for each patch. If a new event falls into a patch and the buffer is full, the older event in popped.
    int eklt_patch_size = 25; // Determines size of patch around corner. All events that fall in this patch are placed into the features buffer.
    int eklt_max_num_iterations = 10; //Maximum number of iterations allowed by the ceres solver to update optical flow direction and warp.

    double eklt_displacement_px = 0; // Controls scaling parameter for batch size calculation: from formula optimal batchsize == 1/Cth * sum |nabla I * v/|v||. displacement_px corresponds to factor 1/Cth
    double eklt_tracking_quality = 0; // minimum tracking quality allowed for a feature to be tracked. Can be a number between 0 (bad track) and 1 (good track). Is a rescaled number computed from ECC cost via tracking_quality == 1 - ecc_cost / 4. Note that ceres returns ecc_cost / 2.
    std::string eklt_bootstrap = ""; // Method for bootstrapping optical flow direction. Options are 'klt', 'events'

    // viewer
    int eklt_update_every_n_events = 20; // Updates viewer data every n events as they are tracked
    double eklt_scale = 4; // Rescaling factor for image. Allows to see subpixel tracking
    double eklt_arrow_length = 5; // Length of optical flow arrow

    bool eklt_display_features = true; // Whether or not to display feature tracks
    bool eklt_display_feature_id = false; // Whether or not to display feature ids
    bool eklt_display_feature_patches = false; // Whether or not to display feature patches

    EkltPatchTimestampAssignment eklt_patch_timestamp_assignment = EkltPatchTimestampAssignment::LATEST_EVENT;

    AsyncFrontendParams eklt_async_frontend_params;


    /**
     * HASTE frontend parameters
     */

    HasteTrackerType haste_tracker_type = HasteTrackerType::HASTE_DIFFERENCE_STAR;
    AsyncFrontendParams haste_async_frontend_params;
    int haste_patch_size = 31;
    int haste_max_corners = 100; // Maximum features allowed to be tracked
    int haste_min_corners = 60; // Minimum features allowed to be tracked
  };

  /**
   * MSCKF-SLAM matrices
   */
  struct MsckfSlamMatrices {
    /**
     * H1 matrix in Li's paper (stacked for all features)
     */
    Matrix H1;

    /**
     * H2 matrix in Li's paper (stacked for all features)
     */
    Matrix H2;

    /**
     * z1 residual vector in Li's paper (stacked for all features)
     */
    Matrix r1;

    /**
     * New SLAM feature vectors (stacked for all features).
     *
     * Features' inverse-depth coordinates, assuming the latest camera as an
     * anchor. These estimates are taken from the MSCKF triangulation prior.
     */
    Matrix features;
  };

  /**
   * Range measurement.
   *
   * Coming from a Laser Range Finder (LRF).
   */
  struct RangeMeasurement
  {
    double timestamp { kInvalid };

    double range { 0.0 };

    /**
     * Image coordinate of the LRF beam.
     *
     * This is assumed to be a single point in the image, i.e. the LRF axis
     * passes by the optical center of the camera.
     */
    Feature img_pt { Feature() };

    /**
     * Normalized image coordinates of the LRF beam.
     *
     * Follows the same assumptions as img_pt.
     */
    Feature img_pt_n { Feature() };
  };

  /**
   * Sun angle measurement.
   *
   * Coming from a sun sensor.
   */
  struct SunAngleMeasurement
  {
    double timestamp { kInvalid };
    double x_angle { 0.0 };
    double y_angle { 0.0 };
  };

  /**
   * VIO update measurement.
   *
   * This struct includes all sensor the measurements that will processed to
   * create an EKF update. This should be at least an image (or a set of visual
   * matches) in xVIO, along side optional additional sensor measurements that
   * are assumed to be synced with the visual data (LRF and sun sensor).
   */
  struct VioMeasurement {
    /**
     * Timestamp.
     *
     * This timestamp is the visual measurement timestamp (camera or match
     * list). Range and sun angle measurement timestamps might different but
     * will be processed at the sam timestamp as a single EKF update.
     */
    double timestamp { kInvalid };

    /**
     * Sequence ID.
     *
     * Consecutively increasing ID associated with the visual measurement
     * (matches or image).
     */
    unsigned int seq { 0 };
    
    /**
     * Visual match measurements.
     *
     * Output of a visual feature tracker (or simulation).
     */
    MatchList matches;

    /**
     * Image measurement.
     *
     * Will only be used if the visual match list struct member has size zero,
     * in which case a feature track will run on that image.
     */
    TiledImage image;

    /**
     * Range measurement.
     */
    RangeMeasurement range;

    /**
     * Sun angle measurement.
     */
    SunAngleMeasurement sun_angle;

    /**
     * Default constructor.
     */
    VioMeasurement() {}

    /**
     * Full constructor.
     */
    VioMeasurement(const double timestamp,
                   const unsigned int seq,
                   const MatchList& matches,
                   const TiledImage& image,
                   const RangeMeasurement& range,
                   const SunAngleMeasurement& sun_angle)
      : timestamp { timestamp }
      , seq { seq }
      , matches { matches }
      , image { image }
      , range { range }
      , sun_angle { sun_angle }
      {}
  };
  
  using Vector3dArray = std::vector<Eigen::Vector3d>;

  using Time = double;
}

#endif // X_VIO_TYPES_H
