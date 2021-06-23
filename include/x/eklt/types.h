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

  struct EkltPatch; //forward decl
  using Patches = std::vector<EkltPatch>; //forward decl

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
