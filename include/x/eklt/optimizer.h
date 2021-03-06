#pragma once

#include <x/eklt/error.h>
#include <x/eklt/eklt_patch.h>
#include <x/eklt/types.h>


namespace x {

  /**
   * @brief The Optimizer performs optimization to find the best warp and optical flow for each patch.
   */
  struct Optimizer {
    explicit Optimizer(Params params, EkltPerformanceLoggerPtr perf_logger = nullptr);

    ~Optimizer();

    /**
     * @brief updates the EKLT parameters
     */
    void setParams(const Params &params);

    void setPerfLogger(const EkltPerformanceLoggerPtr& perf_logger);

    /**
     * @brief Counts how many features are using the current image with timestamp time. 
     * If none are using it, free the memory.
     */
    void decrementCounter(double time);

    /**
     * @ brief precomputes log gradients of the image
     */
    void getLogGradients(const cv::Mat &img, cv::Mat &I_x, cv::Mat &I_y);

    void precomputeLogImageArray(const EkltPatches &patches, const ImageBuffer::iterator &image_it);

    /**
     * @brief perform optimization of cost function (7) in the original paper.
     */
    void optimizeParameters(const cv::Mat &event_frame, EkltPatch &patch, double t);

    Params params_;
    EkltPerformanceLoggerPtr perf_logger_;

    ceres::Problem::Options prob_options;
    ceres::Solver::Options solver_options;

    OptimizerData optimizer_data_;

    int patch_size_;
  };

}
