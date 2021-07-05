#pragma once

#include <x/common/event_types.h>
#include <x/vision/types.h>

#include <deque>
#include <mutex>
#include <fstream>
#include <x/vision/camera.h>
#include <x/eklt/eklt_patch.h>
#include <x/eklt/optimizer.h>
#include <x/eklt/async_feature_tracker.h>
#include <x/eklt/viewer.h>
#include <x/eklt/async_feature_interpolator.h>


namespace x {
/**
 * @brief The EkltTracker class: uses Images to initialize corners and then tracks them using events.
 * Images are subscribed to and, when collected Harris Corners are extracted. Events falling in a patch around the corners
 * forming an event-frame-patch are used as observations and tracked versus the image gradient in the patch
 * at initialization.
 */
  class EkltTracker : public AsyncFeatureTracker {
  public:
    explicit EkltTracker(Camera camera, Viewer &viewer, Params params = {},
                         EventsPerformanceLoggerPtr event_perf_logger = nullptr,
                         EkltPerformanceLoggerPtr eklt_perf_logger = nullptr);

    void setParams(const Params &params) override {
      AsyncFeatureTracker::setParams(params);
      viewer_ptr_->setParams(params);
      optimizer_.setParams(params);
    }


    std::vector<AsyncPatch *> getActivePatches() override {
      std::vector<AsyncPatch *> ret;
      ret.reserve(patches_.size()); // prepare for best case
      for (auto &p : patches_)
        if (!p.lost_)
          ret.push_back(&p);
      return ret;
    }

    void renderVisualization(TiledImage &tracker_debug_image_output);

    /**
     * @brief Initializes a viewer and optimizer with the first image. Also extracts first features.
     * @param image_it CV_8U gray-scale image on which features are extracted.
     */
    void onInit(const ImageBuffer::iterator &image_it) override;

    void onNewImageReceived() override;

    void onPostEvent() override;


    EkltPerformanceLoggerPtr eklt_perf_logger_;
    Optimizer optimizer_;

/**
 * @brief update a patch with the new event, return true if patch position has been updated
 */
bool updatePatch(AsyncPatch &patch, const Event &event) override;

  private:

    /**
     * @brief checks all features if they can be bootstrapped
     */
    void bootstrapAllPossiblePatches(EkltPatches &patches, const ImageBuffer::iterator &image_it);

    /**
   * @brief bootstrapping features: Uses first two frames to initialize feature translation and optical flow.
   */
    void bootstrapFeatureKLT(EkltPatch &patch, const cv::Mat &last_image, const cv::Mat &current_image);

    /**
     * @brief bootstrapping features: Uses first event frame to solve for the best optical flow, given 0 translation.
     */
    void bootstrapFeatureEvents(EkltPatch &patch, const cv::Mat &event_frame);

    /**
     * @brief add new features
     */
    void addFeatures(std::vector<int> &lost_indices, const ImageBuffer::iterator &image_it);

    /**
     * @brief reset patches that have been lost.
     */
    void resetPatches(EkltPatches &new_patches, std::vector<int> &lost_indices, const ImageBuffer::iterator &image_it);

    /**
     * @brief initialize corners on an image
     */
    void initPatches(EkltPatches &patches, std::vector<int> &lost_indices, const int &corners,
                     const ImageBuffer::iterator &image_it);

    inline void padBorders(const cv::Mat &in, cv::Mat &out, int p) {
      out = cv::Mat(in.rows + p * 2, in.cols + p * 2, in.depth());
      cv::Mat gray(out, cv::Rect(p, p, in.cols, in.rows));
      copyMakeBorder(in, out, p, p, p, p, cv::BORDER_CONSTANT);
    }

    /**
     * @brief checks if the optimization cost is above 1.6 (as described in the paper)
     */
    inline bool shouldDiscard(EkltPatch &patch) {
      bool out_of_fov = isPointOutOfView(patch.getCenter());
      bool exceeded_error = patch.tracking_quality_ < params_.eklt_tracking_quality;

      return exceeded_error || out_of_fov;
    }

    inline bool isPointOutOfView(const cv::Point2d &p) const {
      return (p.y < 0 || p.y >= params_.img_height || p.x < 0 || p.x >= params_.img_width);
    }

    /**
     * @brief sets the number of events to process adaptively according to equation (15) in the paper
     */
    void setBatchSize(EkltPatch &patch, const double &d);

    // patch parameters
    EkltPatches patches_;
    std::vector<int> lost_indices_;

    // delegation
    Viewer *viewer_ptr_ = nullptr;

    int viewer_counter_ = 0;

    inline void discardPatch(EkltPatch &patch) {
      // if the patch has been lost record it in lost_indices_
      patch.lost_ = true;
      lost_indices_.push_back(&patch - &patches_[0]);

      if (eklt_perf_logger_)
        eklt_perf_logger_->eklt_tracks_csv.addRow(profiler::now(), patch.getId(), EkltTrackUpdateType::Lost,
                                                  patch.getCurrentTime(), patch.getCenter().x, patch.getCenter().y,
                                                  patch.flow_angle_);
    }

  };

}
