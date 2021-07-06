//
// Created by Florian Mahlknecht on 2021-06-24.
// Copyright (c) 2021 NASA / JPL. All rights reserved.



#include <x/vision/utils.h>
#include <x/eklt/async_feature_interpolator.h>

x::Feature x::AsyncFeatureInterpolator::interpolatePatchToTime(const x::AsyncPatch *p, double t) {
  auto& c_cur = p->getCenter();

  double x, y;

  x = c_cur.x;
  y = c_cur.y;

  if (p->track_hist_.size() >= 2) {
    auto& t_prev = p->track_hist_[p->track_hist_.size() - 2].first;
    auto& c_prev = p->track_hist_[p->track_hist_.size() - 2].second;
    auto& t_cur = p->getCurrentTime();

    switch (params_.ekf_feature_interpolation) {
      case AsyncFrontendFeatureInterpolation::NEAREST_NEIGHBOR: {
        double smallest_deviation = std::numeric_limits<double>::max();
        for (auto it = p->track_hist_.cend() - 1; it >= p->track_hist_.cbegin(); --it) {
          if (smallest_deviation > fabs(it->first - t)) {
            x = it->second.x;
            y = it->second.y;
            smallest_deviation = fabs(it->first - t);
          } else {
            // assume sorted timestamps --> if it gets bigger again, stop
            break;
          }
        }
        break;
      }
      case AsyncFrontendFeatureInterpolation::LINEAR_NO_LIMIT: {
        // time factor assembles normed time distance from current center: e.g. -1 --> previous_center
        double time_factor = 0;

        if (fabs(t_cur - t_prev) >= 1e-9)  // avoid division by zero
          time_factor = (t - t_cur) / (t_cur - t_prev);

        x = c_cur.x + time_factor * (c_cur.x - c_prev.x);
        y = c_cur.y + time_factor * (c_cur.y - c_prev.y);

        break;
      }
      case AsyncFrontendFeatureInterpolation::LINEAR_RELATIVE_LIMIT: {
        // time factor assembles normed time distance from current center: e.g. -1 --> previous_center
        double time_factor = 0;

        if (fabs(t_cur - t_prev) >= 1e-9)  // avoid division by zero
          time_factor = (t - t_cur) / (t_cur - t_prev);

        if (time_factor > 0) {
          time_factor = fmin(time_factor, params_.ekf_feature_extrapolation_limit);
        } else {
          time_factor = fmax(time_factor, -1 - params_.ekf_feature_extrapolation_limit);
        }

        x = c_cur.x + time_factor * (c_cur.x - c_prev.x);
        y = c_cur.y + time_factor * (c_cur.y - c_prev.y);
        break;
      }
      case AsyncFrontendFeatureInterpolation::LINEAR_ABSOLUTE_LIMIT: {
        // time factor assembles normed time distance from current center: e.g. -1 --> previous_center
        double time_factor = 0;

        if (fabs(t_cur - t_prev) >= 1e-9) {  // avoid division by zero
          time_factor = (t - t_cur) / (t_cur - t_prev);

          if (time_factor > 0) {
            time_factor = fmin(time_factor, params_.ekf_feature_extrapolation_limit / (t_cur - t_prev));
          } else {
            time_factor = fmax(time_factor, -1 - params_.ekf_feature_extrapolation_limit / (t_cur - t_prev));
          }
        }

        x = c_cur.x + time_factor * (c_cur.x - c_prev.x);
        y = c_cur.y + time_factor * (c_cur.y - c_prev.y);
        break;
      }

      case AsyncFrontendFeatureInterpolation::NO_INTERPOLATION:
      default:
        x = c_cur.x;
        y = c_cur.y;
        break;
    }
  }

//  // make sure it doesn't get out of bounds just through interpolation EDIT: probably not a good idea
//  x = fmin(x, camera_.getWidth());
//  x = fmax(0, x);
//  y = fmin(y, camera_.getHeight());
//  y = fmax(0, y);

  x::Feature f = createUndistortedFeature(t, x, y);
  return f;
}

x::Feature x::AsyncFeatureInterpolator::createUndistortedFeature(double t, double x, double y) const {
  Feature f {t, 0, x, y, x, y};
  camera_.undistortFeature(f);
  return f;
}

x::MatchList x::AsyncFeatureInterpolator::getMatchListFromPatches(const std::vector<AsyncPatch *>& active_patches) {
  EASY_FUNCTION();

  double interpolation_time = getInterpolationTime(active_patches);

  MatchList matches;

  std::map<int, x::Feature> new_features;

  for (auto& p : active_patches) {
    auto new_pos = interpolatePatchToTime(p, interpolation_time);

    Match m;
    m.current = new_pos;

    bool has_previous_feature = setPreviousFeature(p, m.previous, interpolation_time);

    if (!has_previous_feature | isFeatureOutOfView(m.previous) || isFeatureOutOfView(m.current))
      continue;

    matches.push_back(m);
    new_features[p->getId()] = m.current;
  }

  std::swap(previous_features_, new_features);
  previous_time_ = interpolation_time;

  // remove outliers
  return refineMatches(matches);
}

x::MatchList x::AsyncFeatureInterpolator::refineMatches(x::MatchList &matches) const {
  if (matches.empty() || !params_.enable_outlier_removal)
    return matches;

  std::vector<cv::Point2f> pts1, pts2;
  pts1.reserve(matches.size());
  pts2.reserve(matches.size());

  for (const auto& m : matches) {
    pts1.emplace_back(m.previous.getX(), m.previous.getY());
    pts2.emplace_back(m.current.getX(), m.current.getY());
  }

  auto mask = detectOutliers(pts1, pts2, params_.outlier_method, params_.outlier_param1, params_.outlier_param2);

  MatchList matches_refined;
  matches_refined.reserve(matches.size()); // prepare for best case

  auto m_it = matches.cbegin();

  for (const auto& m : mask) {
    if (m) {
      matches_refined.push_back(*m_it);
    }
    ++m_it;
  }

  return matches_refined;
}

double x::AsyncFeatureInterpolator::getInterpolationTime(const std::vector<AsyncPatch *>& active_patches) const {
  double interpolation_time = std::numeric_limits<double>::lowest();
  int N = 0;

  for (auto& p : active_patches) {
    switch (params_.ekf_update_timestamp) {
      case AsyncFrontendUpdateTimestamp::PATCH_AVERAGE:
        interpolation_time = (N * interpolation_time + p->getCurrentTime()) / (N+1);
        ++N;
        break;
      case AsyncFrontendUpdateTimestamp::PATCH_MAXIMUM:
        interpolation_time = std::max(interpolation_time, p->getCurrentTime());
        break;
    }
  }
  return interpolation_time;
}

bool x::AsyncFeatureInterpolator::setPreviousFeature(const x::AsyncPatch* p, x::Feature& f_prev, double t_cur) {
  bool has_previous_feature = previous_features_.find(p->getId()) != previous_features_.end();

  if (has_previous_feature) {
    f_prev = previous_features_[p->getId()];
  } else {
    if (p->track_hist_.size() < 2)
      return false; // do not convert async features to matches that don't have the necessary history

    if (previous_time_ == kInvalid) {
      // best effort to find proper previous interpolation

      if (params_.ekf_update_strategy == AsyncFrontendUpdateStrategy::EVERY_N_MSEC_WITH_EVENTS) {
        f_prev = interpolatePatchToTime(p, t_cur - params_.ekf_update_every_n * 1e-3);
      } else {
        // take simply first position
        f_prev = createUndistortedFeature(p->track_hist_.front().first, p->track_hist_.front().second.x,
                                              p->track_hist_.front().second.y);
      }
    } else {
      f_prev = interpolatePatchToTime(p, previous_time_);
    }
  }
  return true;
}
