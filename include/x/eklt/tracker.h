#pragma once

#include <x/common/event_types.h>
#include <x/vision/types.h>

#include <deque>
#include <mutex>
#include <fstream>
#include <x/vision/camera.h>

#include "patch.h"
#include "optimizer.h"
#include "viewer.h"


namespace x {
/**
 * @brief The EkltTracker class: uses Images to initialize corners and then tracks them using events.
 * Images are subscribed to and, when collected Harris Corners are extracted. Events falling in a patch around the corners
 * forming an event-frame-patch are used as observations and tracked versus the image gradient in the patch
 * at initialization.
 */
  class EkltTracker {
  public:
    explicit EkltTracker(Camera camera, Viewer &viewer, EkltParams params = {},
                         EkltPerformanceLoggerPtr perf_logger = nullptr);

    /**
     * @brief updates the EKLT parameters in the tracker as well as in the associated viewer and optimizer
     */
    void setParams(const EkltParams& params);

    void setPerfLogger(const EkltPerformanceLoggerPtr& perf_logger);

    void setCamera(const x::Camera& camera) {
      camera_ = camera;
      for (auto& patch : patches_) {
        patch.camera_ptr_ = &camera_;
      }
    }

      /**
     * @brief processes all events in array and returns true if matches have been updated.
     */
    bool processEvents(const EventArray::ConstPtr &msg);

    const MatchList& getMatches() const;

    // TODO why don't we use: current_img.getTimestamp(), current_img.getFrameNumber()
    void processImage(double timestamp, TiledImage &current_img, unsigned int frame_number);

    TiledImage getCurrentImage() {
      return current_image_it_->second;
    }

    void renderVisualization(TiledImage& tracker_debug_image_output);

  private:
    /**
   * @brief Initializes a viewer and optimizer with the first image. Also extracts first features.
   * @param image_it CV_8U gray-scale image on which features are extracted.
   */
    void init(const ImageBuffer::iterator &image_it);

    /**
     * @brief working thread
     */
    void processEvents();

    /**
     * @brief Blocks while there are no events in buffer
     * @param next event
    */
    inline void waitForEvent(Event &ev) {
//      EDIT: make this ROS free
//        static ros::Rate r(100);

      while (true) {
        {
          std::unique_lock<std::mutex> lock(events_mutex_);
          if (events_.empty()) {
            ev = events_.front();
            events_.pop_front();
            return;
          }
        }
//            r.sleep();  // TODO: find alternative (or move to x_vio_ros wrapper)
        VLOG_EVERY_N(1, 30) << "Waiting for events.";
      }
    }

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
    void bootstrapFeatureKLT(Patch &patch, const cv::Mat &last_image, const cv::Mat &current_image);

    /**
     * @brief bootstrapping features: Uses first event frame to solve for the best optical flow, given 0 translation.
     */
    void bootstrapFeatureEvents(Patch &patch, const cv::Mat &event_frame);

    /**
     * @brief add new features
     */
    void addFeatures(std::vector<int> &lost_indices, const ImageBuffer::iterator &image_it);

    /**
     * @brief update a patch with the new event, return true if patch position has been updated
     */
    bool updatePatch(Patch &patch, const Event &event);

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
    inline bool shouldDiscard(Patch &patch) {
      bool out_of_fov = isPointOutOfView(patch.center_);
      bool exceeded_error = patch.tracking_quality_ < params_.tracking_quality;

      return exceeded_error || out_of_fov;
    }

    inline bool isPointOutOfView(const cv::Point2d& p) const {
      return (p.y < 0 || p.y >= sensor_size_.height || p.x < 0 || p.x >= sensor_size_.width);
    }

    /**
     * @brief sets the number of events to process adaptively according to equation (15) in the paper
     */
    void setBatchSize(Patch &patch, const cv::Mat &I_x, const cv::Mat &I_y, const double &d);

    /**
     * @brief Insert an event in the buffer while keeping the buffer sorted
     * This uses insertion sort as the events already come almost always sorted
     */
    inline void insertEventInSortedBuffer(const Event &e) {
      std::unique_lock<std::mutex> lock(events_mutex_);
      events_.push_back(e);
      // insertion sort to keep the buffer sorted
      // in practice, the events come almost always sorted,
      // so the number of iterations of this loop is almost always 0
      auto j = (events_.size() - 1) - 1; // second to last element
      while (j >= 0 && events_[j].ts > e.ts) {
        events_[j + 1] = events_[j];
        j--;
      }
      events_[j + 1] = e;
    }

    Camera camera_;
    EkltParams params_;
    EkltPerformanceLoggerPtr perf_logger_;
    MatchList matches_;
    cv::Size sensor_size_;

    // image flags
    bool got_first_image_;

    // pointers to most recent image and time
    ImageBuffer::iterator current_image_it_;
    double most_current_time_;

    // buffers for images and events
    EventBuffer events_;
    ImageBuffer images_;

    // ros
//    ros::Subscriber event_sub_;
//    image_transport::Subscriber image_sub_;
//    image_transport::ImageTransport it_;
//    ros::NodeHandle nh_;

    // patch parameters
    Patches patches_;
    std::map<int, std::pair<cv::Mat, cv::Mat>> patch_gradients_;
    std::vector<int> lost_indices_;

    // delegation
    Viewer *viewer_ptr_ = NULL;
    Optimizer optimizer_;

    // mutex
    std::mutex events_mutex_;
    std::mutex images_mutex_;

    void updateMatchListFromPatches();

    int viewer_counter_ = 0;

    inline void discardPatch(Patch &patch) {
      // if the patch has been lost record it in lost_indices_
      patch.lost_ = true;
      lost_indices_.push_back(&patch - &patches_[0]);

      if (perf_logger_)
        perf_logger_->tracks_csv.addRow(profiler::now(), patch.id_, EkltTrackUpdateType::Lost,
                                        patch.t_curr_, patch.center_.x, patch.center_.y, patch.flow_angle_);
    }
  };

}
