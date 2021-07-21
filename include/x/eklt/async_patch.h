//
// Created by Florian Mahlknecht on 2021-06-22.
// Copyright (c) 2021 NASA / JPL. All rights reserved.


#include <x/eklt/types.h>
#include <x/vision/types.h>
#include <x/vision/camera.h>

#include <utility>

#pragma once

namespace x {

  class AsyncPatch {

  public:
    // give access to track_history
    friend class AsyncFeatureInterpolator;

    explicit AsyncPatch(EventsPerformanceLoggerPtr perf_logger) : perf_logger_(std::move(perf_logger)) {}
    virtual ~AsyncPatch() = default;

    void resetTrack(double t, const cv::Point2d& center) {
      if (perf_logger_ && track_hist_.size() > 1) {
        // do not log single points as tracks (as they don't generate any kind of updates)
        for (const auto& pair : track_hist_) {
          perf_logger_->event_tracks_csv.addRow(id_, pair.first, pair.second.x, pair.second.y);
        }
      }
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
    EventsPerformanceLoggerPtr perf_logger_;
  };


}