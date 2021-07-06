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
    AsyncFeatureInterpolator(x::AsyncFrontendParams params, x::Camera  cam)
    : camera_(std::move(cam)), params_(params) {}

    Feature interpolatePatchToTime(const AsyncPatch* patch, double t);

    MatchList getMatchListFromPatches(const std::vector<AsyncPatch *>& active_patches,
                                      std::vector<AsyncPatch *>& detected_outliers);

    void setParams(const AsyncFrontendParams& params) {
      params_ = params;
    }

    void setCamera(const x::Camera& camera) {
      camera_ = camera;
    }

  private:
    x::Camera camera_;
    x::AsyncFrontendParams params_;
    std::map<int, x::Feature> previous_features_;
    double previous_time_ = kInvalid;

    inline bool isFeatureOutOfView(const Feature& f) const {
      return (f.getY() < 0 || f.getY() >= camera_.getHeight() || f.getX() < 0 || f.getX() >= camera_.getWidth());
    }

    Feature createUndistortedFeature(double t, double x, double y) const;

    double getInterpolationTime(const std::vector<AsyncPatch *>& active_patches) const;

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




