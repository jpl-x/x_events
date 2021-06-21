//
// Created by Florian Mahlknecht on 2021-03-01.
// Copyright (c) 2021 NASA / JPL. All rights reserved.
//

#include <x/eklt/eklt_vio.h>
#include <x/vio/tools.h>
#include <x/vision/types.h>

#include <x/eklt/tracker.h>

#include <iostream>

// if Boost was compiled with BOOST_NO_EXCEPTIONS defined, it expects a user
// defined trow_exception function, so define a dummy here, if this is the case
#include <exception>
#include <utility>

using namespace x;

namespace boost {
#ifdef BOOST_NO_EXCEPTIONS
  void throw_exception(std::exception const & e) {}; // user defined
#endif
}


EKLTVIO::EKLTVIO(XVioPerformanceLoggerPtr xvio_perf_logger, EkltPerformanceLoggerPtr eklt_perf_logger)
  : ekf_{Ekf(vio_updater_)}
  , msckf_baseline_n_(-1.0)
  , eklt_viewer_()
  , eklt_tracker_(x::Camera(), eklt_viewer_)
  , xvio_perf_logger_(std::move(xvio_perf_logger))
  , eklt_perf_logger_(std::move(eklt_perf_logger)) {
}

bool EKLTVIO::isInitialized() const {
  return initialized_;
}

void EKLTVIO::setUp(const x::Params &params) {
  const x::Camera cam(params.cam_fx, params.cam_fy, params.cam_cx, params.cam_cy, params.cam_distortion_model,
                      params.cam_distortion_parameters, params.img_width, params.img_height);
  const x::Tracker tracker(cam, params.fast_detection_delta, params.non_max_supp, params.block_half_length,
                           params.margin, params.n_feat_min, params.outlier_method, params.outlier_param1,
                           params.outlier_param2);

  // Compute minimum MSCKF baseline in normal plane (square-pixel assumption)
  msckf_baseline_n_ = params.msckf_baseline / (params.img_width * params.cam_fx);

  // Set up tracker and track manager
  const TrackManager track_manager(cam, msckf_baseline_n_, xvio_perf_logger_);
  params_ = params;
  camera_ = cam;
  tracker_ = tracker;
  track_manager_ = track_manager;

  // sets also EKLT params in viewer and optimizer class
  eklt_tracker_.setParams(params);
  eklt_tracker_.setPerfLogger(eklt_perf_logger_);
  eklt_tracker_.setCamera(camera_);

  // Set up EKLTVIO state manager
  const int n_poses_state = params.n_poses_max;
  const int n_features_state = params.n_slam_features_max;
  const StateManager state_manager(n_poses_state, n_features_state);
  state_manager_ = state_manager;

  // Gravity - TODO(jeff) Read from params
  Vector3 g(0.0, 0.0, -9.81);

  // IMU noise
  x::ImuNoise imu_noise;
  imu_noise.n_w = params.n_w;
  imu_noise.n_bw = params.n_bw;
  imu_noise.n_a = params.n_a;
  imu_noise.n_ba = params.n_ba;

  // Updater setup
  x::MatchList matches; // unused empty match list since it's image callback
  TiledImage img;
  vio_updater_ = EkltVioUpdater(state_manager_,
                                track_manager_,
                                params_.sigma_img,
                                params_.sigma_range,
                                params_.rho_0,
                                params_.sigma_rho_0,
                                params_.min_track_length,
                                params_.iekf_iter);

  // EKF setup
  const size_t state_buffer_sz = 250; // TODO(jeff) Read from params
  const State default_state = State(n_poses_state, n_features_state);
  const double a_m_max = 50.0;
  const unsigned int delta_seq_imu = 1;
  const double time_margin_bfr = 0.02;
  ekf_.set(vio_updater_,
           g,
           imu_noise,
           state_buffer_sz,
           default_state,
           a_m_max,
           delta_seq_imu,
           time_margin_bfr);
}

void EKLTVIO::setLastRangeMeasurement(x::RangeMeasurement range_measurement) {
  last_range_measurement_ = range_measurement;
}

void EKLTVIO::setLastSunAngleMeasurement(x::SunAngleMeasurement angle_measurement) {
  last_angle_measurement_ = angle_measurement;
}

State EKLTVIO::processImageMeasurement(double timestamp,
                                       const unsigned int seq,
                                       TiledImage &match_img,
                                       TiledImage &feature_img) {
//  std::cout << "Image update at: " << std::setprecision(17) << timestamp << std::endl;
  // Time correction
  const double timestamp_corrected = timestamp + params_.time_offset;

  // Track features
  auto match_image_tracker_copy = match_img.clone();
  eklt_tracker_.processImage(timestamp_corrected, match_image_tracker_copy, seq);
  //  tracker_.track(match_image_tracker_copy, timestamp_corrected, seq);

  // EKLT does not provide an update from images
  return State();


  MatchList match_list;

  // If we are processing images and last image didn't go back in time
  if (tracker_.checkMatches())
    match_list = tracker_.getMatches();

  // list from a separate tracker module.
  VioMeasurement measurement(timestamp_corrected,
                             seq,
                             match_list,
                             match_img,
                             last_range_measurement_,
                             last_angle_measurement_);

  vio_updater_.setMeasurement(measurement);

  // Process update measurement with xEKF
  State updated_state = ekf_.processUpdateMeasurement();

  // Set state timestamp to original image timestamp for ID purposes in output.
  // We don't do that if that state is invalid, since the timestamp also carries
  // the invalid signature.
  if (updated_state.getTime() != kInvalid)
    updated_state.setTime(timestamp);

  // Populate GUI image outputs
  match_img = match_image_tracker_copy;
  feature_img = vio_updater_.getFeatureImage();

  return updated_state;
}


State EKLTVIO::processEventsMeasurement(const x::EventArray::ConstPtr &events_ptr, TiledImage &tracker_img, TiledImage &feature_img) {
//  std::cout << "Events at timestamps [" << std::setprecision(17) << events_ptr->events.front().ts << ", " << events_ptr->events.back().ts
//            << "] received in xEKLTVIO class." << std::endl;

  auto match_lists_for_ekf_updates = eklt_tracker_.processEvents(events_ptr);

  auto most_recent_state = State();

  double most_recent_timestamp = -1.0;

  for (const auto& matches : match_lists_for_ekf_updates) {
    if (matches.empty())
      continue;

    auto match_img = eklt_tracker_.getCurrentImage().clone();

    const auto timestamp = matches.back().current.getTimestamp();

    const double timestamp_corrected = timestamp + params_.time_offset;

    VioMeasurement measurement(timestamp_corrected,
                               seq_++,
                               matches,
                               match_img,
                               last_range_measurement_,
                               last_angle_measurement_);
    vio_updater_.setMeasurement(measurement);

    // Process update measurement with xEKF
    most_recent_state = ekf_.processUpdateMeasurement();
    most_recent_timestamp = timestamp;
  }

  if(most_recent_state.getTime() != kInvalid) {
    most_recent_state.setTime(most_recent_timestamp);

    // Populate GUI image outputs
    eklt_tracker_.renderVisualization(tracker_img);
    feature_img = vio_updater_.getFeatureImage();
  }

  return most_recent_state;
}

/** Calls the state manager to compute the cartesian coordinates of the SLAM features.
 */
std::vector<Eigen::Vector3d>
EKLTVIO::computeSLAMCartesianFeaturesForState(
  const State &state) {
  return vio_updater_.state_manager_.computeSLAMCartesianFeaturesForState(state);
}

void EKLTVIO::initAtTime(double now) {
  ekf_.lock();
  initialized_ = false;
  vio_updater_.track_manager_.clear();
  vio_updater_.state_manager_.clear();

  // Initial IMU measurement (specific force, velocity)
  Vector3 a_m, w_m;
  a_m << 0.0, 0.0, 9.81;
  w_m << 0.0, 0.0, 0.0;

  // Initial time cannot be 0, which may happen when using simulated Clock time
  // before the first message has been received.
  const double timestamp =
    now > 0.0 ? now : std::numeric_limits<double>::epsilon();

  //////////////////////////////// xEKF INIT ///////////////////////////////////

  // Initial core covariance matrix
  // TODO(jeff) read from params
  const double sigma_dp_x = 0.0;
  const double sigma_dp_y = 0.0;
  const double sigma_dp_z = 0.0;
  const double sigma_dv_x = 0.05;
  const double sigma_dv_y = 0.05;
  const double sigma_dv_z = 0.05;
  const double sigma_dtheta_x = 3.0 * M_PI / 180.0;
  const double sigma_dtheta_y = 3.0 * M_PI / 180.0;
  const double sigma_dtheta_z = 3.0 * M_PI / 180.0;
  const double sigma_dbw_x = 6.0 * M_PI / 180.0;
  const double sigma_dbw_y = 6.0 * M_PI / 180.0;
  const double sigma_dbw_z = 6.0 * M_PI / 180.0;
  const double sigma_dba_x = 0.3;
  const double sigma_dba_y = 0.3;
  const double sigma_dba_z = 0.3;
  const double sigma_dtheta_ic_x = 1.0 * M_PI / 180.0;
  const double sigma_dtheta_ic_y = 1.0 * M_PI / 180.0;
  const double sigma_dtheta_ic_z = 1.0 * M_PI / 180.0;
  const double sigma_dp_ic_x = 0.01;
  const double sigma_dp_ic_y = 0.01;
  const double sigma_dp_ic_z = 0.01;

  const size_t n_poses_state = params_.n_poses_max;
  const size_t n_features_state = params_.n_slam_features_max;

  // Initial vision state estimates and uncertainties are all zero
  const Matrix p_array = Matrix::Zero(n_poses_state * 3, 1);
  const Matrix q_array = Matrix::Zero(n_poses_state * 4, 1);
  const Matrix f_array = Matrix::Zero(n_features_state * 3, 1);
  const Eigen::VectorXd sigma_p_array = Eigen::VectorXd::Zero(n_poses_state * 3);
  const Eigen::VectorXd sigma_q_array = Eigen::VectorXd::Zero(n_poses_state * 3);
  const Eigen::VectorXd sigma_f_array = Eigen::VectorXd::Zero(n_features_state * 3);

  // Construct initial covariance matrix
  const size_t n_err = kSizeCoreErr + n_poses_state * 6 + n_features_state * 3;
  Eigen::VectorXd sigma_diag(n_err);
  sigma_diag << sigma_dp_x, sigma_dp_y, sigma_dp_z,
    sigma_dv_x, sigma_dv_y, sigma_dv_z,
    sigma_dtheta_x, sigma_dtheta_y, sigma_dtheta_z,
    sigma_dbw_x, sigma_dbw_y, sigma_dbw_z,
    sigma_dba_x, sigma_dba_y, sigma_dba_z,
    sigma_dtheta_ic_x, sigma_dtheta_ic_y, sigma_dtheta_ic_z,
    sigma_dp_ic_x, sigma_dp_ic_y, sigma_dp_ic_z,
    sigma_p_array, sigma_q_array, sigma_f_array;

  const Eigen::VectorXd cov_diag = sigma_diag.array() * sigma_diag.array();
  const Matrix cov = cov_diag.asDiagonal();

  // Construct initial state
  const unsigned int dummy_seq = 0;
  State init_state(timestamp,
                   dummy_seq,
                   params_.p,
                   params_.v,
                   params_.q,
                   params_.b_w,
                   params_.b_a,
                   p_array,
                   q_array,
                   f_array,
                   cov,
                   params_.q_ic,
                   params_.p_ic,
                   w_m,
                   a_m);

  // Try to initialize the filter with initial state input
  try {
    ekf_.initializeFromState(init_state);
  } catch (std::runtime_error &e) {
    std::cerr << "bad input: " << e.what() << std::endl;
  } catch (init_bfr_mismatch) {
    std::cerr << "init_bfr_mismatch: the size of dynamic arrays in the "
                 "initialization state match must match the size allocated in "
                 "the buffered states." << std::endl;
  }
  ekf_.unlock();

  initialized_ = true;
}

/** \brief Gets 3D coordinates of MSCKF inliers and outliers.
 *
 *  These are computed in the Measurement class instance.
 */
void EKLTVIO::getMsckfFeatures(x::Vector3dArray &inliers,
                               x::Vector3dArray &outliers) {
  inliers = vio_updater_.getMsckfInliers();
  outliers = vio_updater_.getMsckfOutliers();
}

State EKLTVIO::processImu(const double timestamp,
                          const unsigned int seq,
                          const Vector3 &w_m,
                          const Vector3 &a_m) {
//  std::cout << "IMU update at: " << std::setprecision(17) << timestamp << std::endl;
  return ekf_.processImu(timestamp, seq, w_m, a_m);
}