#pragma once

#include "error.h"
#include "patch.h"
#include "types.h"


namespace x {

  /**
   * @brief The Optimizer performs optimization to find the best warp and optical flow for each patch.
   */
  struct Optimizer {
    explicit Optimizer(EkltParams params);

    ~Optimizer();

    /**
     * @brief updates the EKLT parameters
     */
    void setParams(const EkltParams &params);

    /**
     * @brief Counts how many features are using the current image with timestamp time. 
     * If none are using it, free the memory.
     */
    void decrementCounter(double time);

    /**
     * @ brief precomputes log gradients of the image
     */
    void getLogGradients(const cv::Mat &img, cv::Mat &I_x, cv::Mat &I_y);

    void precomputeLogImageArray(const Patches &patches, const ImageBuffer::iterator &image_it);

    /**
     * @brief perform optimization of cost function (7) in the original paper.
     */
    void optimizeParameters(const cv::Mat &event_frame, Patch &patch);

    EkltParams params_;

    ceres::Problem::Options prob_options;
    ceres::Solver::Options solver_options;
    ceres::LossFunction *loss_function;

    OptimizerData optimizer_data_;

    int patch_size_;
    ceres::CostFunction *cost_function_;
  };

}
