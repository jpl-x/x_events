//
// Created by Florian Mahlknecht on 2021-06-22.
// Copyright (c) 2021 NASA / JPL. All rights reserved.


#include <x/eklt/types.h>
#include <x/vision/types.h>
#include <x/vision/camera.h>

#pragma once

namespace x {

  class AsyncFeatureTrack {
  public:

    AsyncFeatureTrack(const x::Params& params, x::Camera* cam)
    : camera_ptr_(cam) {
      ekf_feature_interpolation_ = params.eklt_ekf_feature_interpolation;
      ekf_feature_extrapolation_limit_ = params.eklt_ekf_feature_extrapolation_limit;
    }

    void setCamera(x::Camera* cam) {
      camera_ptr_ = cam;
    }

    void resetTrack(double t, const cv::Point2d& center) {
      track_hist_.clear();
      track_hist_.emplace_back(t, center);
      previous_feature_ = Feature(t, 0, center.x, center.y, center.x, center.y);
      camera_ptr_->undistortFeature(previous_feature_);
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


    Feature interpolateToTime(double t) const {
      double x, y;

      if (track_hist_.size() < 2) {
        auto &center = getCenter();
        x = center.x;
        y = center.y;
      } else {

        auto& t_prev = track_hist_[track_hist_.size() - 2].first;
        auto& c_prev = track_hist_[track_hist_.size() - 2].second;
        auto& t_cur = getCurrentTime();
        auto& c_cur = getCenter();

        switch (ekf_feature_interpolation_) {
          case EkltEkfFeatureInterpolation::NEAREST_NEIGHBOR:
            if (fabs(t - t_cur) < fabs(t - t_prev)) {
              x = c_cur.x;
              y = c_cur.y;
            } else {
              x = c_prev.x;
              y = c_prev.y;
            }
            break;
          case EkltEkfFeatureInterpolation::LINEAR_NO_LIMIT: {
            // time factor assembles normed time distance from current center: e.g. -1 --> previous_center
            double time_factor = 0;

            if (fabs(t_cur - t_prev) >= 1e-9)  // avoid division by zero
              time_factor = (t - t_cur) / (t_cur - t_prev);

            x = c_cur.x + time_factor * (c_cur.x - c_prev.x);
            y = c_cur.y + time_factor * (c_cur.y - c_prev.y);

            break;
          }
          case EkltEkfFeatureInterpolation::LINEAR_RELATIVE_LIMIT: {
            // time factor assembles normed time distance from current center: e.g. -1 --> previous_center
            double time_factor = 0;

            if (fabs(t_cur - t_prev) >= 1e-9)  // avoid division by zero
              time_factor = (t - t_cur) / (t_cur - t_prev);

            if (time_factor > 0) {
              time_factor = fmin(time_factor, ekf_feature_extrapolation_limit_);
            } else {
              time_factor = fmax(time_factor, -1 - ekf_feature_extrapolation_limit_);
            }

            x = c_cur.x + time_factor * (c_cur.x - c_prev.x);
            y = c_cur.y + time_factor * (c_cur.y - c_prev.y);
            break;
          }
          case EkltEkfFeatureInterpolation::LINEAR_ABSOLUTE_LIMIT: {
            // time factor assembles normed time distance from current center: e.g. -1 --> previous_center
            double time_factor = 0;

            if (fabs(t_cur - t_prev) >= 1e-9) {  // avoid division by zero
              time_factor = (t - t_cur) / (t_cur - t_prev);

              if (time_factor > 0) {
                time_factor = fmin(time_factor, ekf_feature_extrapolation_limit_ / (t_cur - t_prev));
              } else {
                time_factor = fmax(time_factor, -1 - ekf_feature_extrapolation_limit_ / (t_cur - t_prev));
              }
            }

            x = c_cur.x + time_factor * (c_cur.x - c_prev.x);
            y = c_cur.y + time_factor * (c_cur.y - c_prev.y);
            break;
          }

          case EkltEkfFeatureInterpolation::NO_INTERPOLATION:
          default:
            x = c_cur.x;
            y = c_cur.y;
            break;
        }
      }
      return {t, 0, x, y, x, y};
    }

    Match consumeMatch(double t) {
      Match m;
      m.previous = previous_feature_;
      m.current = interpolateToTime(t);
      camera_ptr_->undistortFeature(m.current);
      previous_feature_ = m.current;
      return m;
    }


  private:
    std::vector<std::pair<double, cv::Point2d>> track_hist_;
    Feature previous_feature_;
    x::Camera* camera_ptr_;
    EkltEkfFeatureInterpolation ekf_feature_interpolation_;
    double ekf_feature_extrapolation_limit_;
  };


}