#pragma once

#include <deque>

#include <opencv2/core.hpp>
#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>
#include <x/common/event_types.h>
#include <x/vision/tiled_image.h>
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
#include <easy/profiler.h>



namespace x {

  enum class EkltTrackUpdateType : char {
    Init,
    Bootstrap,
    Update,
    Lost
  };

  std::ostream& operator << (std::ostream& os, const x::EkltTrackUpdateType& obj);
}

namespace x {

  using EventsCsv = CsvWriter<profiler::timestamp_t, profiler::timestamp_t>;
  using OptimizationsCsv = CsvWriter<profiler::timestamp_t, profiler::timestamp_t, int, double>;
  using EKLTTracksCSV = CsvWriter<profiler::timestamp_t, int, EkltTrackUpdateType, double, double, double, double>;


  struct EkltPerformanceLogger {

    explicit EkltPerformanceLogger(const fs::path & path)
     : events_csv(path / "events.csv", {"ts_start", "ts_stop"})
     , optimizations_csv(path / "optimizations.csv", {"ts_start", "ts_stop", "num_iterations", "final_cost"})
     , eklt_tracks_csv(path / "eklt_tracks.csv", {"ts", "id", "update_type", "patch_t_current", "center_x", "center_y", "flow_angle"}) {}

    EventsCsv events_csv;
    OptimizationsCsv optimizations_csv;
    EKLTTracksCSV eklt_tracks_csv;
  };

  typedef std::shared_ptr<EkltPerformanceLogger> EkltPerformanceLoggerPtr;

  enum class EkltEkfFeatureInterpolation : char {
    NO_INTERPOLATION, // always take last
    NEAREST_NEIGHBOR, // take one of both
    LINEAR_NO_LIMIT,  // does linear inter- and extrapolation
    LINEAR_RELATIVE_LIMIT, // does linear inter- and extrapolation till a ratio of interpolation_limit is not exceeded
    LINEAR_ABSOLUTE_LIMIT, // does linear inter- and extrapolation until interpolation_limit in seconds is not exceeded
  };

  enum class EkltEkfUpdateStrategy : char {
    EVERY_ROS_EVENT_MESSAGE,
    EVERY_N_EVENTS,
    // triggers EKF update when incoming event timestamps are more than n msec ahead from previous update
    EVERY_N_MSEC_WITH_EVENTS,
  };

  enum class EkltEkfUpdateTimestamp : char {
    PATCH_AVERAGE,
    PATCH_MAXIMUM,
  };

  struct EkltParams {
    // feature detection
    int max_corners = 100; // Maximum features allowed to be tracked
    int min_corners = 60; // Minimum features allowed to be tracked
    int min_distance = 30; // Minimum distance between detected features. Parameter passed to goodFeatureToTrack
    int block_size = 30; // Block size to compute Harris score. passed to harrisCorner and goodFeaturesToTrack

    // tracker
    double k = 0; // Magic number for Harris score
    double quality_level = 0; // Determines range of harris score allowed between the maximum and minimum. Passed to goodFeaturesToTrack
    double log_eps = 1e-2; // Small constant to compute log image. To avoid numerical issues normally we compute log(img /255 + log_eps)
    double first_image_t = -1; // If specified discards all images until this time.

    // tracker
    int lk_window_size = 15; // Parameter for KLT. Used for bootstrapping feature.
    int num_pyramidal_layers = 2; // Parameter for KLT. Used for bootstrapping feature.
    int batch_size = 200; // Determines the size of the event buffer for each patch. If a new event falls into a patch and the buffer is full, the older event in popped.
    int patch_size = 25; // Determines size of patch around corner. All events that fall in this patch are placed into the features buffer.
    int max_num_iterations = 10; //Maximum number of iterations allowed by the ceres solver to update optical flow direction and warp.

    double displacement_px = 0; // Controls scaling parameter for batch size calculation: from formula optimal batchsize == 1/Cth * sum |nabla I * v/|v||. displacement_px corresponds to factor 1/Cth
    double tracking_quality = 0; // minimum tracking quality allowed for a feature to be tracked. Can be a number between 0 (bad track) and 1 (good track). Is a rescaled number computed from ECC cost via tracking_quality == 1 - ecc_cost / 4. Note that ceres returns ecc_cost / 2.
    std::string bootstrap = ""; // Method for bootstrapping optical flow direction. Options are 'klt', 'events'

    // viewer
    int update_every_n_events = 20; // Updates viewer data every n events as they are tracked
    double scale = 4; // Rescaling factor for image. Allows to see subpixel tracking
    double arrow_length = 5; // Length of optical flow arrow

    bool display_features = true; // Whether or not to display feature tracks
    bool display_feature_id = false; // Whether or not to display feature ids
    bool display_feature_patches = false; // Whether or not to display feature patches

    bool enable_outlier_removal = false; // Whether or not to remove outliers with the same method as xVIO
    int outlier_method = 8; // same as xVIO: 4 for LMEDS, 8 for RANSAC
    double outlier_param1 = 1; // same as xVIO: maximum distance from a non-outlier to an epipolar line in pixels
    double outlier_param2 = 0.99; // same as xVIO: desirable level of confidence e.g. 0.99

    EkltEkfFeatureInterpolation ekf_feature_interpolation = EkltEkfFeatureInterpolation::LINEAR_NO_LIMIT;
    // factor limiting the extrapolation amount. E.g. 0.0 means only interpolation is performed (no extrapolation), 1.0
    // means that at most the time difference between the last two points is used for extrapolation.
    // If < 0, no limit is applied.
    double ekf_feature_extrapolation_limit = -1.0;
    int ekf_update_every_n = -1;

    EkltEkfUpdateStrategy ekf_update_strategy = EkltEkfUpdateStrategy::EVERY_ROS_EVENT_MESSAGE;
    EkltEkfUpdateTimestamp ekf_update_timestamp = EkltEkfUpdateTimestamp::PATCH_AVERAGE;
  };

  struct Patch; //forward decl
  using Patches = std::vector<Patch>; //forward decl

  using EkltEventBuffer = std::deque<Event>;
  using ImageBuffer = std::map<double, TiledImage>;

  struct FeatureTrackData {
    Patches patches;
    double t, t_init;
    cv::Mat image;
  };

  using Grid = ceres::Grid2D<double, 2>;
  using GridPtr = std::unique_ptr<Grid>;
  using Interpolator = ceres::BiCubicInterpolator<Grid>;
  using InterpolatorPtr = std::unique_ptr<ceres::BiCubicInterpolator<Grid>>;

  struct OptimizerDatum {
    OptimizerDatum() {}

    OptimizerDatum(const std::vector<double> &grad, const cv::Mat &img, int num_patches) {
      grad_ = grad;
      grad_grid_ = new Grid(grad_.data(), 0, img.rows, 0, img.cols);
      grad_interp_ = new Interpolator(*grad_grid_);
      ref_counter_ = num_patches;
    }

    void clear() {
      delete grad_interp_;
      delete grad_grid_;
    }

    std::vector<double> grad_;
    Grid *grad_grid_;
    Interpolator *grad_interp_;
    int ref_counter_;
  };

  using OptimizerData = std::map<double, OptimizerDatum>;

}



