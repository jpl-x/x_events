#pragma once

#include <x/common/event_types.h>
#include <x/vision/types.h>

#include <deque>
#include <mutex>
#include <fstream>
#include <x/vision/camera.h>

#include "eklt_patch.h"
#include "optimizer.h"
#include "viewer.h"
#include "async_patch.h"
#include "async_feature_interpolator.h"


namespace x {
/**
 * @brief The EkltTracker class: uses Images to initialize corners and then tracks them using events.
 * Images are subscribed to and, when collected Harris Corners are extracted. Events falling in a patch around the corners
 * forming an event-frame-patch are used as observations and tracked versus the image gradient in the patch
 * at initialization.
 */
  class EkltTracker {
  public:
    explicit EkltTracker(Camera camera, Viewer &viewer, Params params = {},
                         EkltPerformanceLoggerPtr perf_logger = nullptr);

    /**
     * @brief updates the EKLT parameters in the tracker as well as in the associated viewer and optimizer
     */
    void setParams(const Params &params);

    void setPerfLogger(const EkltPerformanceLoggerPtr &perf_logger);

    void setCamera(const x::Camera &camera) {
      interpolator_.setCamera(camera);
    }


    std::vector<AsyncPatch *> getActivePatches() {
      std::vector<AsyncPatch *> ret;
      ret.reserve(patches_.size()); // prepare for best case
      for (auto &p : patches_)
        if (!p.lost_)
          ret.push_back(&p);
      return ret;
    }

    /**
   * @brief processes all events in array and returns true if matches have been updated.
   */
    std::vector<MatchList> processEvents(const EventArray::ConstPtr &msg);

    /**
     * Processes new image. Timestamp is passed instead of using current_img.getTimestamp() to allow for corrections.
     * @param timestamp corrected timestamp
     * @param current_img APS frame
     */
    void processImage(double timestamp, TiledImage &current_img);

    TiledImage getCurrentImage() {
      return current_image_it_->second;
    }

    void renderVisualization(TiledImage &tracker_debug_image_output);

  private:
    /**
   * @brief Initializes a viewer and optimizer with the first image. Also extracts first features.
   * @param image_it CV_8U gray-scale image on which features are extracted.
   */
    void init(const ImageBuffer::iterator &image_it);

    /**
    * @brief Always assigns image to the first image before time  t_start
    */
    inline bool updateFirstImageBeforeTime(double t_start, ImageBuffer::iterator &current_image_it) {
      bool next_image = false;
      auto next_image_it = current_image_it;

      while (next_image_it->first < t_start) {
        ++next_image_it;
        if (next_image_it == images_.end())
          break;

        if (next_image_it->first < t_start) {
          next_image = true;
          current_image_it = next_image_it;
        }
      }

      return next_image;
    }

    /**
     * @brief checks all features if they can be bootstrapped
     */
    void bootstrapAllPossiblePatches(Patches &patches, const ImageBuffer::iterator &image_it);

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
     * @brief update a patch with the new event, return true if patch position has been updated
     */
    bool updatePatch(AsyncPatch &patch, const Event &event);

    /**
     * @brief reset patches that have been lost.
     */
    void resetPatches(Patches &new_patches, std::vector<int> &lost_indices, const ImageBuffer::iterator &image_it);

    /**
     * @brief initialize corners on an image
     */
    void initPatches(Patches &patches, std::vector<int> &lost_indices, const int &corners,
                     const ImageBuffer::iterator &image_it);

    /**
     * @brief extract patches
     */
    void extractPatches(Patches &patches, const int &num_patches, const ImageBuffer::iterator &image_it);

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
    void setBatchSize(EkltPatch &patch, const cv::Mat &I_x, const cv::Mat &I_y, const double &d);

    Params params_;
    EkltPerformanceLoggerPtr perf_logger_;

    // image flags
    bool got_first_image_;
    int events_till_next_ekf_update_ = -1;
    double last_ekf_update_timestamp_ = -1;

    // pointers to most recent image and time
    ImageBuffer::iterator current_image_it_;
    double most_current_time_;

    ImageBuffer images_;

    // patch parameters
    Patches patches_;
    std::map<int, std::pair<cv::Mat, cv::Mat>> patch_gradients_;
    std::vector<int> lost_indices_;

    // delegation
    Viewer *viewer_ptr_ = nullptr;
    Optimizer optimizer_;
    AsyncFeatureInterpolator interpolator_;

    int viewer_counter_ = 0;

    inline void discardPatch(EkltPatch &patch) {
      // if the patch has been lost record it in lost_indices_
      patch.lost_ = true;
      lost_indices_.push_back(&patch - &patches_[0]);

      if (perf_logger_)
        perf_logger_->eklt_tracks_csv.addRow(profiler::now(), patch.getId(), EkltTrackUpdateType::Lost,
                                             patch.getCurrentTime(), patch.getCenter().x, patch.getCenter().y,
                                             patch.flow_angle_);
    }
  };

}
