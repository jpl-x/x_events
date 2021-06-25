#include <thread>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <iomanip>
#include <utility>

#include <x/eklt/eklt_tracker.h>

#include <cassert>
#include <easy/profiler.h>
#include <x/vision/camera.h>
#include <x/vision/utils.h>


using namespace x;

EkltTracker::EkltTracker(Camera camera, Viewer &viewer, Params params, EkltPerformanceLoggerPtr p_logger)
  : params_(std::move(params)), perf_logger_(std::move(p_logger))
  , sensor_size_(0, 0), got_first_image_(false), most_current_time_(-1.0)
  , viewer_ptr_(&viewer), optimizer_(params_, perf_logger_)
  , interpolator_(params, camera) {
}

void EkltTracker::setParams(const Params &params) {
  params_ = params;
  viewer_ptr_->setParams(params);
  optimizer_.setParams(params);
  interpolator_.setParams(params);

  if (params_.eklt_ekf_update_strategy == EkltEkfUpdateStrategy::EVERY_N_EVENTS) {
    events_till_next_ekf_update_ = params_.eklt_ekf_update_every_n;
  }
}

void EkltTracker::initPatches(Patches &patches, std::vector<int> &lost_indices, const int &corners,
                              const ImageBuffer::iterator &image_it) {
  // extract Harris corners
  extractPatches(patches, corners, image_it);

  // fill up patches to full capacity and set all of the filled patches to lost
  for (int i = patches.size(); i < corners; i++) {
    patches.emplace_back(params_);
    lost_indices.push_back(i);
  }

  // extract log image gradient patches that act as features and are used to
  // compute the adaptive batchSize as per equation 15
  const int &p = (params_.eklt_patch_size - 1) / 2;
  cv::Mat I_x, I_y, I_x_padded, I_y_padded;
  optimizer_.getLogGradients(image_it->second, I_x, I_y);

  padBorders(I_x, I_x_padded, p);
  padBorders(I_y, I_y_padded, p);

  for (size_t i = 0; i < patches.size(); i++) {
    EkltPatch &patch = patches[i];
    if (patch.lost_) {
      patch_gradients_[i] = std::make_pair(cv::Mat::zeros(2 * p + 1, 2 * p + 1, CV_64F),
                                           cv::Mat::zeros(2 * p + 1, 2 * p + 1, CV_64F));
    } else {
      const int x_min = patch.getCenter().x;
      const int y_min = patch.getCenter().y;

      patch_gradients_[i] = std::make_pair(
        I_x_padded.rowRange(y_min, y_min + 2 * p + 1).colRange(x_min, x_min + 2 * p + 1).clone(),
        I_y_padded.rowRange(y_min, y_min + 2 * p + 1).colRange(x_min, x_min + 2 * p + 1).clone());

      // sets adaptive batch size based on gradients according to equation (15) in the paper
      setBatchSize(patch, patch_gradients_[i].first, patch_gradients_[i].second, params_.eklt_displacement_px);
    }
  }
}

void EkltTracker::init(const ImageBuffer::iterator &image_it) {
  // extracts corners and extracts patches around them.
  initPatches(patches_, lost_indices_, params_.eklt_max_corners, image_it);

  // initializes the image gradients in x and y directions for the first image
  // and initializes the ceres cubic interpolator for use in the optimizer
  optimizer_.precomputeLogImageArray(patches_, image_it);

  // init the data arrays that are used by the viewer
  // the time stamp drawn on the viewer is zeroed at image_it->first
  if (params_.eklt_display_features)
    viewer_ptr_->initViewData(image_it->first);
}

bool EkltTracker::updatePatch(EkltPatch &patch, const Event &event) {
  // if patch is lost or event does not fall within patch
  // or the event has occurred before the most recent patch timestamp
  // or the patch has not been bootstrapped yet, do not process event
  if (patch.lost_ ||
      (params_.eklt_bootstrap == "klt" && !patch.initialized_) ||
      !patch.contains(event.x, event.y) ||
      patch.getCurrentTime() > event.ts)
    return false;

  patch.insert(event);

  // start optimization if there are update_rate new events in the patch
//  int update_rate = std::min<int>(patch.update_rate_, patch.batch_size_); // EDIT from patch.event_counter_ < update_rate
  if (patch.event_buffer_.size() < patch.batch_size_ || patch.event_counter_ < patch.batch_size_)
    return false;

  // compute event frame according to equation (2) in the paper
  cv::Mat event_frame;
  auto event_accumulation_timestamp = patch.getEventFrame(event_frame);

  // bootstrap using the events
  if (!patch.initialized_ && params_.eklt_bootstrap == "events")
    bootstrapFeatureEvents(patch, event_frame);

  // update feature position and recompute the adaptive batchsize

  switch (params_.eklt_patch_timestamp_assignment) {
    case EkltPatchTimestampAssignment::LATEST_EVENT:
      optimizer_.optimizeParameters(event_frame, patch, event.ts);
      break;
    case EkltPatchTimestampAssignment::ACCUMULATED_EVENTS_CENTER:
      optimizer_.optimizeParameters(event_frame, patch, event_accumulation_timestamp);
      break;
  }

  if (perf_logger_)
    perf_logger_->eklt_tracks_csv.addRow(profiler::now(), patch.getId(), EkltTrackUpdateType::Update,
                                         patch.getCurrentTime(), patch.getCenter().x, patch.getCenter().y, patch.flow_angle_);

  setBatchSize(patch, patch_gradients_[&patch - &patches_[0]].first, patch_gradients_[&patch - &patches_[0]].second,
               params_.eklt_displacement_px);

  if (shouldDiscard(patch)) {
    discardPatch(patch);
  }

  // if we arrived here, the patch has been updated
  return true;
}

void EkltTracker::addFeatures(std::vector<int> &lost_indices, const ImageBuffer::iterator &image_it) {
  // find new patches to replace them lost features
  Patches patches;
  extractPatches(patches, lost_indices.size(), image_it);

  if (!patches.empty()) {
    // pass the new image to the optimizer to use for future optimizations
    optimizer_.precomputeLogImageArray(patches, image_it);

    // reset all lost features with newly initialized ones
    resetPatches(patches, lost_indices, image_it);
  }
}

void EkltTracker::bootstrapAllPossiblePatches(Patches &patches, const ImageBuffer::iterator &image_it) {
  for (size_t i = 0; i < patches.size(); ++i) {
    EkltPatch &patch = patches[i];

    // if a patch is already bootstrapped, lost or has just been extracted
    if (patch.initialized_ || patch.lost_ || patch.t_init_ == image_it->first)
      continue;

    // perform bootstrapping using KLT and the first 2 frames, and compute the adaptive batch size
    // with the newly found parameters
    bootstrapFeatureKLT(patch, images_[patch.t_init_], image_it->second);
    setBatchSize(patch, patch_gradients_[i].first, patch_gradients_[i].second, params_.eklt_displacement_px);
  }
}

void EkltTracker::setBatchSize(EkltPatch &patch, const cv::Mat &I_x, const cv::Mat &I_y, const double &d) {
  // implements the equation (15) of the paper
  cv::Mat gradient = d * std::cos(patch.flow_angle_) * I_x + d * std::sin(patch.flow_angle_) * I_y;
  patch.batch_size_ = std::min<double>(cv::norm(gradient, cv::NORM_L1), params_.eklt_batch_size);
  patch.batch_size_ = std::max<int>(5, patch.batch_size_);
}

void
EkltTracker::resetPatches(Patches &new_patches, std::vector<int> &lost_indices, const ImageBuffer::iterator &image_it) {
  const int &p = (params_.eklt_patch_size - 1) / 2;
  cv::Mat I_x, I_y, I_x_padded, I_y_padded;

  optimizer_.getLogGradients(image_it->second, I_x, I_y);
  padBorders(I_x, I_x_padded, p);
  padBorders(I_y, I_y_padded, p);

  for (int i = new_patches.size() - 1; i >= 0; i--) {
    int index = lost_indices[i];
    // for each lost feature decrement the ref counter of the optimizer (will free image gradients when no more
    // features use the image with timestamp patches_[index].t_init_
    optimizer_.decrementCounter(patches_[index].t_init_);

    // reset lost patches with new ones
    EkltPatch &reset_patch = new_patches[i];
    patches_[index].reset(reset_patch.getCenter(), reset_patch.t_init_);
    lost_indices.erase(lost_indices.begin() + i);

    // reinitialize the image gradients of new features
    const int x_min = reset_patch.getCenter().x - p;
    const int y_min = reset_patch.getCenter().y - p;
    cv::Mat p_I_x = I_x_padded.rowRange(y_min, y_min + 2 * p + 1).colRange(x_min, x_min + 2 * p + 1);
    cv::Mat p_I_y = I_y_padded.rowRange(y_min, y_min + 2 * p + 1).colRange(x_min, x_min + 2 * p + 1);
    patch_gradients_[index] = std::make_pair(p_I_x.clone(), p_I_y.clone());
    setBatchSize(reset_patch, patch_gradients_[index].first, patch_gradients_[index].second, params_.eklt_displacement_px);
  }
}

void EkltTracker::extractPatches(Patches &patches, const int &num_patches, const ImageBuffer::iterator &image_it) {
  std::vector<cv::Point2d> features;

  // mask areas which are within a distance min_distance of other features or along the border.
  int hp = (params_.eklt_patch_size - 1) / 2;
  int h = sensor_size_.height;
  int w = sensor_size_.width;
  cv::Mat mask = cv::Mat::ones(sensor_size_, CV_8UC1);
  mask.rowRange(0, hp).colRange(0, w - 1).setTo(0);
  mask.rowRange(h - hp, h - 1).colRange(0, w - 1).setTo(0);
  mask.rowRange(0, h - 1).colRange(0, hp).setTo(0);
  mask.rowRange(0, h - 1).colRange(w - hp, w - 1).setTo(0);

  const int &min_distance = params_.eklt_min_distance;
  for (EkltPatch &patch: patches_) {
    if (patch.lost_) continue;

    double min_x = std::fmax(patch.getCenter().x - min_distance, 0);
    double max_x = std::fmin(patch.getCenter().x + min_distance, w - 1);
    double min_y = std::fmax(patch.getCenter().y - min_distance, 0);
    double max_y = std::fmin(patch.getCenter().y + min_distance, h - 1);
    mask.rowRange(min_y, max_y).colRange(min_x, max_x).setTo(0);
  }

  // extract harris corners which are suitable
  // since they correspond to strong edges which also generate alot of events.
  VLOG(2) << "Harris corner detector with N=" << num_patches << " quality=" << params_.eklt_quality_level
          << " min_dist=" << params_.eklt_min_distance << " block_size=" << params_.eklt_block_size << " k=" << params_.eklt_k
          << " image_depth=" << image_it->second.depth() << " mask_ratio="
          << cv::sum(mask)[0] / (mask.cols * mask.rows);

  cv::goodFeaturesToTrack(image_it->second, features, num_patches,
                          params_.eklt_quality_level,
                          params_.eklt_min_distance, mask,
                          params_.eklt_block_size,
                          true,
                          params_.eklt_k);

  // initialize patches centered at the features with an initial pixel warp
  VLOG(1)
  << "Extracted " << features.size() << " new features on image at t=" << std::setprecision(15) << image_it->first
  << " s.";
  for (const auto & feature : features) {
    patches.emplace_back(feature, image_it->first, params_);
    EkltPatch &patch = patches[patches.size() - 1];
    if (perf_logger_)
      perf_logger_->eklt_tracks_csv.addRow(profiler::now(), patch.getId(), EkltTrackUpdateType::Init,
                                           patch.getCurrentTime(), patch.getCenter().x, patch.getCenter().y, patch.flow_angle_);
  }
}

void EkltTracker::bootstrapFeatureKLT(EkltPatch &patch, const cv::Mat &last_image, const cv::Mat &current_image) {
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
  double opt_flow_angle = std::atan2(next_points[0].y - points[0].y, next_points[0].x - points[0].x);
  patch.flow_angle_ = opt_flow_angle;

  // initialize warping as pure translation to new point
  patch.warping_.at<double>(0, 2) = -(next_points[0].x - points[0].x);
  patch.warping_.at<double>(1, 2) = -(next_points[0].y - points[0].y);
  patch.updateCenter(current_image_it_->first);

  // check if new patch has been lost due to leaving the fov
  bool should_discard = bool(patch.getCenter().y < 0 || patch.getCenter().y >= sensor_size_.height || patch.getCenter().x < 0 ||
                             patch.getCenter().x >= sensor_size_.width);
  if (should_discard) {
    patch.lost_ = true;
    lost_indices_.push_back(&patch - &patches_[0]);
  } else {
    patch.initialized_ = true;
    if (perf_logger_)
      perf_logger_->eklt_tracks_csv.addRow(profiler::now(), patch.getId(), EkltTrackUpdateType::Bootstrap,
                                           patch.getCurrentTime(), patch.getCenter().x, patch.getCenter().y, patch.flow_angle_);
  }
}

void EkltTracker::bootstrapFeatureEvents(EkltPatch &patch, const cv::Mat &event_frame) {
  // Implement a bootstrapping mechanism for computing the optical flow direction via
  // \nabla I \cdot v= - \Delta E --> v = - \nabla I ^ \dagger \Delta E (assuming no translation or rotation
  // of the feature.
  int index = &patch - &patches_[0];

  cv::Mat &I_x = patch_gradients_[index].first;
  cv::Mat &I_y = patch_gradients_[index].second;

  double s_I_xx = cv::sum(I_x.mul(I_x))[0];
  double s_I_yy = cv::sum(I_y.mul(I_y))[0];
  double s_I_xy = cv::sum(I_x.mul(I_y))[0];
  double s_I_xt = cv::sum(I_x.mul(event_frame))[0];
  double s_I_yt = cv::sum(I_y.mul(event_frame))[0];

  cv::Mat M = (cv::Mat_<double>(2, 2) << s_I_xx, s_I_xy, s_I_xy, s_I_yy);
  cv::Mat b = (cv::Mat_<double>(2, 1) << s_I_xt, s_I_yt);

  cv::Mat v = -M.inv() * b;

  patch.flow_angle_ = std::atan2(v.at<double>(0, 0), v.at<double>(1, 0));
  patch.initialized_ = true;
}

class EventPerfHelper {
public:
  explicit EventPerfHelper(const EkltPerformanceLoggerPtr& ptr) {
    if (ptr) {
      perf_logger_ = ptr;
      t_start_ = profiler::now();
    }
  }

  ~EventPerfHelper() {
    if (perf_logger_) {
      perf_logger_->events_csv.addRow(t_start_, profiler::now());
    }
  }

  EkltPerformanceLoggerPtr perf_logger_;
  profiler::timestamp_t t_start_;
};

std::vector<MatchList> EkltTracker::processEvents(const EventArray::ConstPtr &msg) {
  std::vector<MatchList> match_lists_for_ekf_updates;

  if (!got_first_image_) {
    LOG_EVERY_N(INFO, 20) << "Events dropped since no image present.";
    return match_lists_for_ekf_updates;
  }

  bool did_some_patch_change = false;
  for (const auto &ev : msg->events) {
    EASY_EVENT("Single Event");
    EventPerfHelper helper(perf_logger_);
    // keep track of the most current time with latest time stamp from event
    if (ev.ts >= most_current_time_)
      most_current_time_ = ev.ts;
    else if (fabs(ev.ts - most_current_time_) > 1e-6)  // if order wrong and spaced more than 1us
      LOG(WARNING) << "Processing event behind most current time: " << std::setprecision(15) << ev.ts << " < " << most_current_time_ << ". Events might not be in order!";

    int num_features_tracked = patches_.size();
    // go through each patch and update the event frame with the new event
    for (EkltPatch &patch: patches_) {
      did_some_patch_change |= updatePatch(patch, ev);
      // count tracked features
      if (patch.lost_)
        num_features_tracked--;
    }

    if (updateFirstImageBeforeTime(most_current_time_, current_image_it_)) // enter if new image found
    {
      // bootstrap patches that need to be due to new image
      if (params_.eklt_bootstrap == "klt")
        bootstrapAllPossiblePatches(patches_, current_image_it_);

      // replenish features if there are too few
      if (lost_indices_.size() > static_cast<size_t>(params_.eklt_max_corners - params_.eklt_min_corners))
        addFeatures(lost_indices_, current_image_it_);

      // erase old image
      auto image_it = current_image_it_;
      image_it--;
      images_.erase(image_it);
    }

    switch (params_.eklt_ekf_update_strategy) {
      case EkltEkfUpdateStrategy::EVERY_ROS_EVENT_MESSAGE:
        // nothing to do here
        break;
      case EkltEkfUpdateStrategy::EVERY_N_EVENTS:
        if (--events_till_next_ekf_update_ <= 0) {
          if (did_some_patch_change) {
            events_till_next_ekf_update_ = params_.eklt_ekf_update_every_n;
            did_some_patch_change = false;
            match_lists_for_ekf_updates.push_back(interpolator_.getMatchListFromPatches(getActivePatches()));
          } else {
            events_till_next_ekf_update_ = 1; // try on next event again
          }
        }
        break;
      case EkltEkfUpdateStrategy::EVERY_N_MSEC_WITH_EVENTS:
        if (ev.ts - last_ekf_update_timestamp_ >= params_.eklt_ekf_update_every_n * 1e-3 && did_some_patch_change) {
          did_some_patch_change = false;
          last_ekf_update_timestamp_ = ev.ts;
          match_lists_for_ekf_updates.push_back(interpolator_.getMatchListFromPatches(getActivePatches()));
        }
        break;
    }

    if (params_.eklt_display_features && ++viewer_counter_ % params_.eklt_update_every_n_events == 0)
      viewer_ptr_->setViewData(patches_, most_current_time_, current_image_it_);
  }

  if (did_some_patch_change && params_.eklt_ekf_update_strategy == EkltEkfUpdateStrategy::EVERY_ROS_EVENT_MESSAGE) {
    match_lists_for_ekf_updates.push_back(interpolator_.getMatchListFromPatches(getActivePatches()));
  }
  return match_lists_for_ekf_updates;
}

void EkltTracker::processImage(double timestamp, TiledImage &current_img, unsigned int frame_number) {
  // tiling not implemented
  assert(current_img.getNTilesH() == 1 && current_img.getNTilesW() == 1);

  if (sensor_size_.width <= 0)
    sensor_size_ = cv::Size(current_img.getTileWidth(),
                            current_img.getTileHeight());
  images_.insert(std::make_pair(timestamp, current_img.clone()));

  if (!got_first_image_) {
    VLOG(1) << "Found first image.";
    current_image_it_ = images_.begin();
    most_current_time_ = current_image_it_->first;
    init(current_image_it_);
    got_first_image_ = true;
  }
}


void EkltTracker::renderVisualization(TiledImage &tracker_debug_image_output) {
  viewer_ptr_->renderView();
  tracker_debug_image_output = viewer_ptr_->getFeatureTrackViewImage();
}

void EkltTracker::setPerfLogger(const EkltPerformanceLoggerPtr &perf_logger) {
  perf_logger_ = perf_logger;
  optimizer_.setPerfLogger(perf_logger);
}
