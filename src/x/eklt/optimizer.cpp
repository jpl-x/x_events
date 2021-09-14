
#include <x/eklt/optimizer.h>
#include <easy/profiler.h>

#include <utility>
#include <x/eklt/utils.h>


using namespace x;

Optimizer::Optimizer(Params params, EkltPerformanceLoggerPtr perf_logger)
  : params_(std::move(params)), perf_logger_(std::move(perf_logger)), patch_size_(params_.eklt_patch_size) {
  // EDIT: let ceres take ownership of these objects function to avoid leaks / double frees
//  prob_options.cost_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
//  prob_options.local_parameterization_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
//  prob_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;

  solver_options.minimizer_progress_to_stdout = false;
  solver_options.num_threads = 1;
  solver_options.linear_solver_type = ceres::DENSE_QR;
  solver_options.logging_type = ceres::SILENT;
  solver_options.max_num_iterations = params_.eklt_max_num_iterations;
  solver_options.use_nonmonotonic_steps = true;
}


void Optimizer::setParams(const Params &params) {
  params_ = params;
}

Optimizer::~Optimizer() {
  for (auto &it : optimizer_data_)
    it.second.clear();
}

void Optimizer::decrementCounter(double time) {
  // keep track of reference counter to release gradient data
  if (optimizer_data_.find(time) == optimizer_data_.end())
    return;

  --optimizer_data_[time].ref_counter_;
  if (optimizer_data_[time].ref_counter_ == 0) {
    optimizer_data_[time].clear();
    optimizer_data_.erase(time);
  }
}

void Optimizer::getLogGradients(const cv::Mat &img, cv::Mat &I_x, cv::Mat &I_y) {
  computeLogImgGradients(img, I_x, I_y, params_.eklt_log_eps, params_.eklt_use_linlog_scale);
}

void Optimizer::precomputeLogImageArray(const EkltPatches &patches, const ImageBuffer::iterator &image_it) {
  cv::Mat I_x, I_y;
  getLogGradients(image_it->second, I_x, I_y);

  std::vector<double> grad;

  // initialize grid that is used by ceres interpolator
  for (int row = 0; row < image_it->second.rows; row++) {
    for (int col = 0; col < image_it->second.cols; col++) {
      grad.push_back(I_x.at<double>(row, col));
      grad.push_back(I_y.at<double>(row, col));
    }
  }

  // store this for later use
  double t = image_it->first;
  optimizer_data_[t] = OptimizerDatum(grad, image_it->second, patches.size());
}

void Optimizer::optimizeParameters(const cv::Mat &event_frame, EkltPatch &patch, double t) {
  auto start = profiler::now();
//  EASY_FUNCTION();

  ceres::Problem problem(prob_options);
  ceres::Solver::Summary summary;

  // for now 3 free parameters for warp, x, y translation and theta rotation
  double p0[3];
  ErrorRotation::getP0(p0, patch.warping_);
  double v0[] = {patch.flow_angle_};

  // create cost functor that depends on the event frame, and current and initial
  // patch location
  auto cost_function = Generator::Create(patch.getCenter(),
                                     patch.init_center_,
                                     &event_frame,
                                     optimizer_data_[patch.t_init_].grad_interp_);

  problem.AddResidualBlock(cost_function, nullptr, p0, v0);

  ceres::Solve(solver_options, &problem, &summary);

  // update patch according to new optimizer
  cv::Mat camera_warp;
  ErrorRotation::getWarp(p0, camera_warp);

  // update state of patch
  patch.warping_ = camera_warp.clone();
  patch.flow_angle_ = fmod(v0[0], 2 * M_PI);

  //remap tracking cost to 0-1 (1 is best and 0 is bad)
  patch.tracking_quality_ = 1 - summary.final_cost / 2;
  patch.updateCenter(t);

  if (perf_logger_)
    perf_logger_->optimizations_csv.addRow(start, profiler::now(), summary.iterations.size(), summary.final_cost);
}

void Optimizer::setPerfLogger(const EkltPerformanceLoggerPtr &perf_logger) {
  perf_logger_ = perf_logger;
}


