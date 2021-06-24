//
// Created by Florian Mahlknecht on 2021-06-22.
// Copyright (c) 2021 NASA / JPL. All rights reserved.


#include <x/eklt/types.h>
#include <x/vision/types.h>
#include <x/vision/camera.h>

#pragma once

namespace x {

  class AsyncPatch {

  public:
    // give access to track_history
    friend class AsyncFeatureInterpolator;

    AsyncPatch() = default;

    void resetTrack(double t, const cv::Point2d& center) {
      track_hist_.clear();
      track_hist_.emplace_back(t, center);
    }

    void updateTrack(double t, const cv::Point2d &center) {
      track_hist_.emplace_back(t, center);
    }

    const cv::Point2d & getCenter() const {
      return track_hist_.back().second;
    }

    const double& getCurrentTime() const {
      return track_hist_.back().first;
    }

    int getId() const { return id_; }

  protected:
    int id_ {-1};
    void assignNewId() {
      static int id = 0;
      id_ = id++;
    }

  private:
    std::vector<std::pair<double, cv::Point2d>> track_hist_;
  };


}