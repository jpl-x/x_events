//
// Created by Florian Mahlknecht on 2021-07-05.
// Copyright (c) 2021 NASA / JPL. All rights reserved.

#include <x/haste/haste_tracker.h>
#include <x/vision/camera.h>
#include <x/vision/utils.h>

#include <opencv2/core/core.hpp>
#include <opencv2/video/tracking.hpp>

#include <easy/profiler.h>
#include <x/haste/tracking/correlation_tracker.hpp>
#include <x/haste/tracking/haste_correlation_tracker.hpp>
#include <x/haste/tracking/haste_correlation_star_tracker.hpp>
#include <x/haste/tracking/haste_difference_tracker.hpp>
#include <x/haste/tracking/haste_difference_star_tracker.hpp>
#include <x/eklt/utils.h>

using namespace x;

HasteTracker::HasteTracker(Camera camera, Params params, EventsPerformanceLoggerPtr event_perf_logger)
  : AsyncFeatureTracker(std::move(camera), params.haste_async_frontend_params, std::move(event_perf_logger))
  , params_(std::move(params)) {}

void HasteTracker::initPatches(HastePatches &patches, std::vector<int> &lost_indices, const int &corners,
                              const ImageBuffer::iterator &image_it) {

  std::vector<cv::Point2d> features;
  // extract Harris corners
  extractFeatures(features, corners, image_it, params_.haste_patch_size);

  for (const auto & feature : features) {
    patches.emplace_back(feature, image_it->first, params_, event_perf_logger_);
    patches.back().hypothesis_tracker_ = createHypothesisTracker(image_it->first, feature.x, feature.y, 0);
  }

  // fill up patches to full capacity and set all of the filled patches to lost
  for (int i = patches.size(); i < corners; i++) {
    patches.emplace_back(params_, event_perf_logger_);
    lost_indices.push_back(i);
  }

//  // extract log image gradient patches that act as features and are used to
//  // compute the adaptive batchSize as per equation 15
//  const int &p = (params_.eklt_patch_size - 1) / 2;
//  cv::Mat I_x, I_y, I_x_padded, I_y_padded;
//  optimizer_.getLogGradients(image_it->second, I_x, I_y);
//
//  padBorders(I_x, I_x_padded, p);
//  padBorders(I_y, I_y_padded, p);
//
//  for (size_t i = 0; i < patches.size(); i++) {
//    HastePatch &patch = patches[i];
//    if (patch.lost_) {
//      patch.gradient_xy_ = std::make_pair(cv::Mat::zeros(2 * p + 1, 2 * p + 1, CV_64F),
//                                          cv::Mat::zeros(2 * p + 1, 2 * p + 1, CV_64F));
//    } else {
//      const int x_min = patch.getCenter().x;
//      const int y_min = patch.getCenter().y;
//
//      patch.gradient_xy_ = std::make_pair(
//        I_x_padded.rowRange(y_min, y_min + 2 * p + 1).colRange(x_min, x_min + 2 * p + 1).clone(),
//        I_y_padded.rowRange(y_min, y_min + 2 * p + 1).colRange(x_min, x_min + 2 * p + 1).clone());
//
//      // sets adaptive batch size based on gradients according to equation (15) in the paper
//      setBatchSize(patch, params_.eklt_displacement_px);
//    }
//  }
}

void HasteTracker::onInit(const ImageBuffer::iterator &image_it) {
  // extracts corners and extracts patches around them.
  initPatches(patches_, lost_indices_, params_.haste_max_corners, image_it);

//  // initializes the image gradients in x and y directions for the first image
//  // and initializes the ceres cubic interpolator for use in the optimizer
//  optimizer_.precomputeLogImageArray(patches_, image_it);
//
//  // init the data arrays that are used by the viewer
//  // the time stamp drawn on the viewer is zeroed at image_it->first
//  if (params_.eklt_display_features)
//    viewer_ptr_->initViewData(image_it->first);
}

bool HasteTracker::updatePatch(AsyncPatch &async_patch, const Event &event) {
  auto& patch = dynamic_cast<HastePatch&>(async_patch);
  // if patch is lost or the event has occurred before the most recent patch timestamp do not process event
  if (patch.lost_ || patch.getCurrentTime() > event.ts)
    return false;

  auto update_type = patch.hypothesis_tracker_->pushEvent(event.ts, event.x, event.y);

  // push event to accumulate features, but don't go any further
  if (params_.haste_bootstrap_from_frames && !patch.initialized_)
    return false;

  switch (update_type) {

    case haste::HypothesisPatchTracker::kOutOfRange:
    case haste::HypothesisPatchTracker::kInitializingEvent:
      // skip
      return false;
    case haste::HypothesisPatchTracker::kRegularEvent:
      // confirm last position
      patch.confirmLastPosition(patch.hypothesis_tracker_->t());
      break;
    case haste::HypothesisPatchTracker::kStateEvent:
      patch.updateCenter(patch.hypothesis_tracker_->t(), {patch.hypothesis_tracker_->x(), patch.hypothesis_tracker_->y()});
      break;
  }

  if (shouldDiscard(patch)) {
    discardPatch(patch);
  }

  // if we arrived here, the patch has been updated
  return true;
}

void HasteTracker::addFeatures(std::vector<int> &lost_indices, const ImageBuffer::iterator &image_it) {
  // find new patches to replace them lost features
  std::vector<cv::Point2d> features;

  extractFeatures(features, lost_indices.size(), image_it, params_.haste_patch_size);

  if (!features.empty()) {
    HastePatches patches;

    for (const auto & feature : features) {
      patches.emplace_back(feature, image_it->first, params_, event_perf_logger_);
      patches.back().hypothesis_tracker_ = createHypothesisTracker(image_it->first, feature.x, feature.y, 0);
    }

//    // pass the new image to the optimizer to use for future optimizations
//    optimizer_.precomputeLogImageArray(patches, image_it);

    // reset all lost features with newly initialized ones
    resetPatches(patches, lost_indices, image_it);
  }
}

void HasteTracker::bootstrapAllPossiblePatches(HastePatches &patches, const ImageBuffer::iterator &image_it) {
  std::vector<double> grad;
  Grid* grid = nullptr;
  Interpolator* interpolator = nullptr;

  for (auto & patch : patches) {
    // if a patch is already bootstrapped, lost or has just been extracted
    if (patch.initialized_ || patch.lost_ || patch.t_init_ == image_it->first)
      continue;

    // compute gradients only if necessary
    if (!interpolator) {
      cv::Mat I_x, I_y;
      computeLogImgGradients(image_it->second, I_x, I_y, 1e-2, true);

      // initialize grid that is used by ceres interpolator
      for (int row = 0; row < image_it->second.rows; row++) {
        for (int col = 0; col < image_it->second.cols; col++) {
          grad.push_back(I_x.at<double>(row, col));
          grad.push_back(I_y.at<double>(row, col));
        }
      }
      grid = new Grid(grad.data(), 0, image_it->second.rows, 0, image_it->second.cols);
      interpolator = new Interpolator(*grid);
    }

    // perform bootstrapping using KLT and the first 2 frames
    bootstrapFeatureKLT(patch, images_[patch.t_init_], image_it->second, interpolator);
  }

  delete grid;
  delete interpolator;
}

void HasteTracker::resetPatches(HastePatches &new_patches, std::vector<int> &lost_indices, const ImageBuffer::iterator &image_it) {
  const int &p = (params_.haste_patch_size - 1) / 2;
  cv::Mat I_x, I_y, I_x_padded, I_y_padded;

//  optimizer_.getLogGradients(image_it->second, I_x, I_y);
  padBorders(I_x, I_x_padded, p);
  padBorders(I_y, I_y_padded, p);

  for (int i = new_patches.size() - 1; i >= 0; i--) {
    int index = lost_indices[i];
    // for each lost feature decrement the ref counter of the optimizer (will free image gradients when no more
    // features use the image with timestamp patches_[index].t_init_
//    optimizer_.decrementCounter(patches_[index].t_init_);

    // reset lost patches with new ones
    HastePatch &reset_patch = new_patches[i];
    // reinitialize the image gradients of new features
//    const int x_min = reset_patch.getCenter().x - p;
//    const int y_min = reset_patch.getCenter().y - p;
//    cv::Mat p_I_x = I_x_padded.rowRange(y_min, y_min + 2 * p + 1).colRange(x_min, x_min + 2 * p + 1);
//    cv::Mat p_I_y = I_y_padded.rowRange(y_min, y_min + 2 * p + 1).colRange(x_min, x_min + 2 * p + 1);

//    auto grad = std::make_pair(p_I_x.clone(), p_I_y.clone());

    patches_[index].reset(reset_patch.getCenter(), reset_patch.t_init_);
    patches_[index].hypothesis_tracker_ = createHypothesisTracker(reset_patch.t_init_, reset_patch.init_center_.x, reset_patch.init_center_.y, 0);
//    setBatchSize(patches_[index], params_.eklt_displacement_px);
//
    lost_indices.erase(lost_indices.begin() + i);
  }
}

void HasteTracker::bootstrapFeatureKLT(HastePatch &patch, const cv::Mat &last_image, const TiledImage &current_image, Interpolator* interpolator) {
  // bootstrap feature by initializing its warp and optical flow with KLT on successive images
  std::vector<cv::Point2f> points = {patch.init_center_};
  std::vector<cv::Point2f> next_points;

  // track feature for one frame
  std::vector<float> error;
  std::vector<uchar> status;
  cv::Size window(params_.eklt_lk_window_size, params_.eklt_lk_window_size);
  cv::calcOpticalFlowPyrLK(last_image, current_image, points, next_points, status, error, window,
                           params_.eklt_num_pyramidal_layers);

  // compute optical flow angle as direction where the feature moved
  double klt_delta_x = next_points[0].x - points[0].x;
  double klt_delta_y = next_points[0].y - points[0].y;
  double opt_flow_angle = std::atan2(klt_delta_y, klt_delta_x);
//  patch.flow_angle_ = opt_flow_angle;

//  // initialize warping as pure translation to new point
//  patch.warping_.at<double>(0, 2) = -(next_points[0].x - points[0].x);
//  patch.warping_.at<double>(1, 2) = -(next_points[0].y - points[0].y);
  patch.updateCenter(current_image_it_->first, next_points[0]);

  // check if new patch has been lost due to leaving the fov
  bool should_discard = bool(patch.getCenter().y < 0 || patch.getCenter().y >= params_.img_height || patch.getCenter().x < 0 ||
                             patch.getCenter().x >= params_.img_width);
  if (should_discard) {
    patch.lost_ = true;
    lost_indices_.push_back(&patch - &patches_[0]);
  } else {
//    // this could  be used as "outlier rejection" - if the patch diverged already from the KLT estimate
//    double delta_x = fabs(next_points[0].x - patch.hypothesis_tracker_->x());
//    double delta_y = fabs(next_points[0].y - patch.hypothesis_tracker_->y());
//    double error = sqrt(delta_x* delta_x + delta_y*delta_y);
//    if (error > 5) {
//      std::cerr << "Patch completely off already: HASTE: " << patch.hypothesis_tracker_->x() << ", " << patch.hypothesis_tracker_->y() << " KLT: " << next_points[0] << std::endl;
//    }
    const auto &[et, ex, ey] =  patch.hypothesis_tracker_->event_window().middleEvent();
    haste::CorrelationTracker::Hypothesis bootstrapped_hypothesis {et, next_points[0].x, next_points[0].y, 0};

    double v_m;
    if (params_.haste_bootstrap_with_unit_length_of) {
      v_m = 1;
    } else {
      double distance = sqrt(klt_delta_x*klt_delta_x + klt_delta_y*klt_delta_y);
      v_m = distance / (current_image.getTimestamp() - patch.t_init_);
    }

    haste::HypothesisPatchTracker::Patch new_template;


    // same code as in EKLT error
    for (size_t y=0; y< haste::HypothesisPatchTracker::kPatchSize; ++y) {
      for (size_t  x=0; x<haste::HypothesisPatchTracker::kPatchSize; ++x) {
        double x_p = next_points[0].x + static_cast<double>(x) - haste::HypothesisPatchTracker::kPatchSizeHalf;
        double y_p = next_points[0].y + static_cast<double>(y) - haste::HypothesisPatchTracker::kPatchSizeHalf;

        // g for gradient
        double g[2];
        interpolator->Evaluate(y_p, x_p, g);

        new_template(y, x) = v_m * cos(opt_flow_angle) * g[0] +  v_m * sin(opt_flow_angle) * g[1];
      }
    }

    new_template = ((new_template - new_template.minCoeff()) / (new_template.maxCoeff() - new_template.minCoeff())).matrix();

    patch.hypothesis_tracker_->initializeTrackerWithTemplate(new_template, bootstrapped_hypothesis);
    patch.initialized_ = true;
  }
}


HypothesisTrackerPtr HasteTracker::createHypothesisTracker(double t, double x, double y, double theta) {
  switch (params_.haste_tracker_type) {
    case HasteTrackerType::CORRELATION:
      return std::make_shared<haste::CorrelationTracker>(t, x, y, theta);
    case HasteTrackerType::HASTE_CORRELATION:
      return std::make_shared<haste::HasteCorrelationTracker>(t, x, y, theta);
    case HasteTrackerType::HASTE_CORRELATION_STAR:
      return std::make_shared<haste::HasteCorrelationStarTracker>(t, x, y, theta);
    case HasteTrackerType::HASTE_DIFFERENCE:
      return std::make_shared<haste::HasteDifferenceTracker>(t, x, y, theta);
    case HasteTrackerType::HASTE_DIFFERENCE_STAR:
    default:
      return std::make_shared<haste::HasteDifferenceStarTracker>(t, x, y, theta);
  }
}


void HasteTracker::onNewImageReceived() {
  if (params_.haste_bootstrap_from_frames) {
    // bootstrap patches that need to be due to new image
    bootstrapAllPossiblePatches(patches_, current_image_it_);
  }

  // replenish features if there are too few
  if (lost_indices_.size() > static_cast<size_t>(params_.haste_max_corners - params_.haste_min_corners))
    addFeatures(lost_indices_, current_image_it_);
}

void HasteTracker::discardPatch(AsyncPatch &async_patch) {
  HastePatch& patch = dynamic_cast<HastePatch&>(async_patch);
  // if the patch has been lost record it in lost_indices_
  patch.lost_ = true;
  lost_indices_.push_back(&patch - &patches_[0]);
}



