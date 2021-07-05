//
// Created by Florian Mahlknecht on 2021-07-05.
// Copyright (c) 2021 NASA / JPL. All rights reserved.


#pragma once


#include <x/eklt/async_patch.h>


namespace x {

/**
 * @brief The HastePatch struct which corresponds to an image patch defined by its center (feature position) and
 * dimension (patch size)
 */
  struct HastePatch : public AsyncPatch {

    HastePatch(const cv::Point2d &center, double t_init, const x::Params &params, const EventsPerformanceLoggerPtr& perf_logger)
      : AsyncPatch(perf_logger)
      , init_center_(center)
      , t_init_(t_init)
      , lost_(false)
      , initialized_(false) {

      patch_size_ = params.eklt_patch_size;
      half_size_ = (params.eklt_patch_size - 1) / 2;

      reset(init_center_, t_init);
    }

    explicit HastePatch(const x::Params &params, const EventsPerformanceLoggerPtr& perf_logger) : HastePatch(cv::Point2f(-1, -1), -1, params, perf_logger) {
      // contstructor for initializing lost features
      lost_ = true;
    }

    /**
     * @brief contains checks if event is contained in square around the current feature position
     */
    inline bool contains(double x, double y) const {
      return half_size_ >= std::abs(x - getCenter().x) && half_size_ >= std::abs(y - getCenter().y);
    }

    /**
     * @brief checks if 2x2 ((x,y) to (x+1,y+1)) update is within patch boundaries
     */
    inline bool contains_patch_update(int x, int y) const {
      return (((x + 1) < patch_size_) && (x >= 0) &&
              ((y + 1) < patch_size_) && (y >= 0));
    }

    inline void updateCenter(double t, const cv::Point2d& new_center) {
//      warpPixel(init_center_, new_center);
      updateTrack(t, new_center);
    }

    /**
     * @brief resets patch after it has been lost.
     */
    inline void reset(const cv::Point2d &init_center, double t) {
      // reset feature after it has been lost
      lost_ = false;
      initialized_ = false;

      resetTrack(t, init_center);

      init_center_ = init_center;
      t_init_ = t;

      assignNewId();
    }

    cv::Point2d init_center_;

    int patch_size_;
    int half_size_;

    double t_init_;

    bool lost_;
    bool initialized_;
  };

}
