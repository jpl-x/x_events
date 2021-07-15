//
// Created by Florian Mahlknecht on 2021-07-15.
// Copyright (c) 2021 NASA / JPL. All rights reserved.


#include <x/vio/tools.h>
#include <iostream>

using namespace x;

x::DistortionModel x::stringToDistortionModel(std::string &dist_model) {
  std::transform(dist_model.begin(), dist_model.end(), dist_model.begin(), ::tolower);
  x::DistortionModel m = DistortionModel::FOV;
  if (dist_model == "fov") {
    m = DistortionModel::FOV;
  } else if (dist_model == "radial-tangential") {
    m = DistortionModel::RADIAL_TANGENTIAL;
  } else if (dist_model == "equidistant") {
    m = DistortionModel::EQUIDISTANT;
  } else {
    std::cout << "ERROR: Distortion model '%s' not recognized, assuming FOV" << dist_model.c_str() << std::endl;
  }
  return m;
}

x::AsyncFrontendFeatureInterpolation x::stringToInterpolationStrategy(std::string &strategy) {
  std::transform(strategy.begin(), strategy.end(), strategy.begin(), ::tolower);
  x::AsyncFrontendFeatureInterpolation s = x::AsyncFrontendFeatureInterpolation::LINEAR_NO_LIMIT;

  if (strategy == "none") {
    s = x::AsyncFrontendFeatureInterpolation::NO_INTERPOLATION;
  } else if (strategy == "nearest-neighbor") {
    s = x::AsyncFrontendFeatureInterpolation::NEAREST_NEIGHBOR;
  } else if (strategy == "linear-no-limit") {
    s = x::AsyncFrontendFeatureInterpolation::LINEAR_NO_LIMIT;
  } else if (strategy == "linear-relative-limit") {
    s = x::AsyncFrontendFeatureInterpolation::LINEAR_RELATIVE_LIMIT;
  } else if (strategy == "linear-absolute-limit") {
    s = x::AsyncFrontendFeatureInterpolation::LINEAR_ABSOLUTE_LIMIT;
  } else {
    std::cout << "ERROR: Interpolation strategy '%s' not recognized, assuming linear-no-limit" << strategy.c_str() << std::endl;
  }

  return s;
}

x::AsyncFrontendUpdateTimestamp x::stringToEkfUpdateTimestamp(std::string &ekf_update_timestamp) {
  std::transform(ekf_update_timestamp.begin(), ekf_update_timestamp.end(), ekf_update_timestamp.begin(), ::tolower);
  auto s = AsyncFrontendUpdateTimestamp::PATCH_AVERAGE;

  if (ekf_update_timestamp == "patches-average") {
    s = x::AsyncFrontendUpdateTimestamp::PATCH_AVERAGE;
  } else if (ekf_update_timestamp == "patches-maximum") {
    s = x::AsyncFrontendUpdateTimestamp::PATCH_MAXIMUM;
  } else if (ekf_update_timestamp == "latest-event-ts") {
    s = x::AsyncFrontendUpdateTimestamp::LATEST_EVENT_TS;
  } else {
    std::cout << "ERROR: EkltEkfUpdateTimestamp '%s' not recognized, assuming patches-average" << ekf_update_timestamp.c_str() << std::endl;
  }

  return s;
}

x::AsyncFrontendUpdateStrategy x::stringToEkfUpdateStrategy(std::string &ekf_update_strategy) {
  std::transform(ekf_update_strategy.begin(), ekf_update_strategy.end(), ekf_update_strategy.begin(), ::tolower);
  auto s = AsyncFrontendUpdateStrategy::EVERY_ROS_EVENT_MESSAGE;

  if (ekf_update_strategy == "every-ros-event-message") {
    s = x::AsyncFrontendUpdateStrategy::EVERY_ROS_EVENT_MESSAGE;
  } else if (ekf_update_strategy == "every-n-events") {
    s = x::AsyncFrontendUpdateStrategy::EVERY_N_EVENTS;
  }  else if (ekf_update_strategy == "every-n-msec-with-events") {
    s = x::AsyncFrontendUpdateStrategy::EVERY_N_MSEC_WITH_EVENTS;
  } else {
    std::cout << "ERROR: EkltEkfUpdateStrategy '%s' not recognized, assuming every-ros-event-message" << ekf_update_strategy.c_str() << std::endl;
  }

  return s;
}

x::EkltPatchTimestampAssignment x::stringToEkltPatchTimestampAssignment(std::string& timestamp_assignment) {
  std::transform(timestamp_assignment.begin(), timestamp_assignment.end(), timestamp_assignment.begin(), ::tolower);
  auto s = EkltPatchTimestampAssignment::LATEST_EVENT;

  if (timestamp_assignment == "latest-event") {
    s = x::EkltPatchTimestampAssignment::LATEST_EVENT;
  } else if (timestamp_assignment == "accumulated-events-center") {
    s = x::EkltPatchTimestampAssignment::ACCUMULATED_EVENTS_CENTER;
  } else {
    std::cout << "ERROR: EkltPatchTimestampAssignment '%s' not recognized, assuming latest-event" << timestamp_assignment.c_str() << std::endl;
  }

  return s;
}

x::HasteTrackerType x::stringToHasteTrackerType(std::string &haste_tracker_type) {
  std::transform(haste_tracker_type.begin(), haste_tracker_type.end(), haste_tracker_type.begin(), ::tolower);
  auto s = HasteTrackerType::HASTE_DIFFERENCE_STAR;

  if (haste_tracker_type == "correlation") {
    s = x::HasteTrackerType::CORRELATION;
  } else if (haste_tracker_type == "haste-correlation") {
    s = x::HasteTrackerType::HASTE_CORRELATION;
  } else if (haste_tracker_type == "haste-correlation-star") {
    s = x::HasteTrackerType::HASTE_CORRELATION_STAR;
  } else if (haste_tracker_type == "haste-difference") {
    s = x::HasteTrackerType::HASTE_DIFFERENCE;
  } else if (haste_tracker_type == "haste-difference-star") {
    s = x::HasteTrackerType::HASTE_DIFFERENCE_STAR;
  } else {
    std::cout << "ERROR: HasteTrackerType '%s' not recognized, assuming haste-difference-star" << haste_tracker_type.c_str() << std::endl;
  }

  return s;
}

