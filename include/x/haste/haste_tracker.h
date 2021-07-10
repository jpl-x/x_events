//
// Created by Florian Mahlknecht on 2021-07-05.
// Copyright (c) 2021 NASA / JPL. All rights reserved.


#pragma once

#include <x/common/event_types.h>
#include <x/vision/types.h>
#include <x/vision/camera.h>
#include <x/eklt/async_feature_tracker.h>
#include <x/haste/haste_patch.h>


namespace x {

  class HasteTracker : public AsyncFeatureTracker {
  public:
    explicit HasteTracker(Camera camera, Params params = {},
                         EventsPerformanceLoggerPtr event_perf_logger = nullptr);

    std::vector<AsyncPatch *> getActivePatches() override {
      std::vector<AsyncPatch *> ret;
      ret.reserve(patches_.size()); // prepare for best case
      for (auto &p : patches_)
        if (!p.lost_)
          ret.push_back(&p);
      return ret;
    }

//    void renderVisualization(TiledImage &tracker_debug_image_output);

    /**
     * @brief Initializes a viewer and optimizer with the first image. Also extracts first features.
     * @param image_it CV_8U gray-scale image on which features are extracted.
     */
    void onInit(const ImageBuffer::iterator &image_it) override;

    void onNewImageReceived() override;


    /**
     * @brief update a patch with the new event, return true if patch position has been updated
     */
    bool updatePatch(AsyncPatch &patch, const Event &event) override;

    void setParams(const Params &params) {
      params_ = params;
      setAsyncFrontendParams(params.haste_async_frontend_params);
    }

  private:

    /**
     * @brief checks all features if they can be bootstrapped
     */
    void bootstrapAllPossiblePatches(HastePatches &patches, const ImageBuffer::iterator &image_it);

    /**
   * @brief bootstrapping features: Uses first two frames to initialize feature translation and optical flow.
   */
    void bootstrapFeatureKLT(HastePatch &patch, const cv::Mat &last_image, const TiledImage &current_image, Interpolator* interpolator);

    HypothesisTrackerPtr createHypothesisTracker(double t, double x, double y, double theta);

    /**
     * @brief add new features
     */
    void addFeatures(std::vector<int> &lost_indices, const ImageBuffer::iterator &image_it);

    /**
     * @brief reset patches that have been lost.
     */
    void resetPatches(HastePatches &new_patches, std::vector<int> &lost_indices, const ImageBuffer::iterator &image_it);

    /**
     * @brief initialize corners on an image
     */
    void initPatches(HastePatches &patches, std::vector<int> &lost_indices, const int &corners,
                     const ImageBuffer::iterator &image_it);

    inline void padBorders(const cv::Mat &in, cv::Mat &out, int p) {
      out = cv::Mat(in.rows + p * 2, in.cols + p * 2, in.depth());
      cv::Mat gray(out, cv::Rect(p, p, in.cols, in.rows));
      copyMakeBorder(in, out, p, p, p, p, cv::BORDER_CONSTANT);
    }


    /**
     * @brief checks if the optimization cost is above 1.6 (as described in the paper)
     */
    inline bool shouldDiscard(HastePatch &patch) {
      bool out_of_fov = isPointOutOfView(patch.getCenter());
//      bool exceeded_error = patch.tracking_quality_ < params_.eklt_tracking_quality;

      return out_of_fov;
    }

    inline bool isPointOutOfView(const cv::Point2d &p) const {
      return (p.y < 0 || p.y >= params_.img_height || p.x < 0 || p.x >= params_.img_width);
    }

    Params params_;

    // patch parameters
    HastePatches patches_;
    std::vector<int> lost_indices_;

    void discardPatch(AsyncPatch& async_patch) override;

  };

}
