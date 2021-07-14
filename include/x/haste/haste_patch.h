//
// Created by Florian Mahlknecht on 2021-07-05.
// Copyright (c) 2021 NASA / JPL. All rights reserved.


#pragma once


#include <x/eklt/async_patch.h>

#include <utility>
#include <x/haste/types.h>


namespace x {

/**
 * @brief The HastePatch struct which corresponds to an image patch defined by its center (feature position) and
 * dimension (patch size)
 */
  struct HastePatch : public AsyncPatch {

    HastePatch(cv::Point2d center, double t_init, const x::Params &params, const EventsPerformanceLoggerPtr& perf_logger)
      : AsyncPatch(perf_logger)
      , init_center_(std::move(center))
      , t_init_(t_init)
      , lost_(false)
      , initialized_(false) {

      patch_size_ = params.haste_patch_size;
      half_size_ = (params.haste_patch_size - 1) / 2;

      reset(init_center_, t_init);
    }

    explicit HastePatch(const x::Params &params, const EventsPerformanceLoggerPtr& perf_logger) : HastePatch(cv::Point2f(-1, -1), -1, params, perf_logger) {
      // contstructor for initializing lost features
      lost_ = true;
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
      hypothesis_tracker_.reset();

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

    /**
     * Nomenclature of HASTE is not compatible with the one we use in X. A tracker indeed tracks just a single patch in
     * HASTE --> our HastePatch therefore owns one hypothesis_tracker.
     */
    x::HypothesisTrackerPtr hypothesis_tracker_;
  };

}
