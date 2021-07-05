//
// Created by Florian Mahlknecht on 2021-07-05.
// Copyright (c) 2021 NASA / JPL. All rights reserved.

#pragma once

#include <x/eklt/async_patch.h>
#include <vector>
#include <x/common/event_types.h>
#include <x/vision/types.h>
#include <x/vision/camera.h>
#include <x/eklt/async_feature_interpolator.h>


namespace x {

  class AsyncFeatureTracker {
  public:

    explicit AsyncFeatureTracker(Camera camera, Params params, EventsPerformanceLoggerPtr event_perf_logger = nullptr);

    virtual bool updatePatch(AsyncPatch &patch, const Event &event) = 0;

    virtual void onInit(const ImageBuffer::iterator &image_it) = 0;

    virtual std::vector<AsyncPatch *> getActivePatches() = 0;

    virtual ~AsyncFeatureTracker() = default;

    virtual void onNewImageReceived() {};

    virtual void onPostEvent() {};

    /**
     * @brief updates the EKLT parameters in the tracker as well as in the associated viewer and optimizer
     */
    virtual void setParams(const Params &params);

    void setCamera(const Camera &camera) {
      interpolator_.setCamera(camera);
    }

    /**
     * @brief processes all events in array and returns a list of match lists, able to generate an EKF update
     */
    std::vector<MatchList> processEvents(const EventArray::ConstPtr &msg);

    /**
     * Processes new image. Timestamp is passed instead of using current_img.getTimestamp() to allow for corrections.
     * @param timestamp corrected timestamp
     * @param current_img APS frame
     */
    void processImage(double timestamp, const TiledImage &current_img);

    void extractFeatures(std::vector<cv::Point2d> &features, int num_patches, const ImageBuffer::iterator &image_it);

    TiledImage getCurrentImage() {
      return current_image_it_->second;
    }

  protected:
    // pointers to most recent image and time
    ImageBuffer::iterator current_image_it_;

    // image flags
    bool got_first_image_;
    ImageBuffer images_;
    AsyncFeatureInterpolator interpolator_;
    double most_current_time_;
    Params params_;

    EventsPerformanceLoggerPtr event_perf_logger_;


  private:
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

    int events_till_next_ekf_update_ = -1;
    double last_ekf_update_timestamp_ = -1;
  };

}


