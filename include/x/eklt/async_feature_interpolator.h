//
// Created by Florian Mahlknecht on 2021-06-24.
// Copyright (c) 2021 NASA / JPL. All rights reserved.



#pragma once


#include <x/vision/camera.h>
#include <x/vio/types.h>

#include <utility>
#include <map>
#include "async_patch.h"


namespace x {

  class AsyncFeatureInterpolator {
  public:
    AsyncFeatureInterpolator(x::AsyncFrontendParams params, x::Camera  cam, EventsPerformanceLoggerPtr event_perf_logger)
    : camera_(std::move(cam)), params_(params), event_perf_logger_(std::move(event_perf_logger)) {}

    Feature interpolatePatchToTime(const AsyncPatch* patch, double t);

    MatchList getMatchListFromPatches(const std::vector<AsyncPatch *>& active_patches,
                                      std::vector<AsyncPatch *>& detected_outliers, double latest_event_ts);

    void setParams(const AsyncFrontendParams& params) {
      params_ = params;
    }

    void setCamera(const x::Camera& camera) {
      camera_ = camera;
    }

  private:
    x::Camera camera_;
    x::AsyncFrontendParams params_;
    EventsPerformanceLoggerPtr event_perf_logger_;
    std::map<int, x::Feature> previous_features_;
    /**
     * Minimum time gap between consecutive interpolation timestamps (ensures monotonically increasing update timestamps)
     */
    const double time_eps_ = 1e-6;
    double previous_time_ = kInvalid;

    inline bool isFeatureOutOfView(const Feature& f) const {
      return (f.getY() < 0 || f.getY() >= camera_.getHeight() || f.getX() < 0 || f.getX() >= camera_.getWidth());
    }

    Feature createUndistortedFeature(double t, double x, double y) const;

    double getInterpolationTime(const std::vector<AsyncPatch *>& active_patches, double latest_event_ts) const;

    /**
     * Assigns previous feature to f_prev, if found for patch p. Under nominal conditions this is simply reading from
     * the map previous_features_. If it's not found there interpolation or first feature position are used, unless
     * feature history is too small, in which case false is returned.
     * @param p
     * @param f_prev
     * @param t_cur current timestamp we generate features from
     * @return true on success, false otherwise
     */
    bool setPreviousFeature(const x::AsyncPatch* p, x::Feature& f_prev, double t_cur);
  };


}




