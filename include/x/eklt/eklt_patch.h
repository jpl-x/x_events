#pragma once

#include <mutex>
#include <random>
#include <deque>

#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <x/eklt/error.h>
#include <x/eklt/types.h>

#include <easy/profiler.h>

#include <x/eklt/async_patch.h>


namespace x {

/**
 * @brief The EkltPatch struct which corresponds to an image patch defined by its center (feature position) and
 * dimension (patch size)
 */
  struct EkltPatch : public AsyncPatch {
    /**
     * @brief EkltPatch constructor for patch. Is defined by its center, linear warping matrix, half size ( half the patch
     * side length and the direction of optical flow in its region.
     * @param center Position of the patch defined by the location of the associated corner
     * @param warping Affine 3x3 matrix describing the current transformation taking pixels in the patch to the initial image.
     * @param flow_angle Angle (rad) of the optical flow in the patch around the initial feature
     * @param half_size half size of the patch side length
     * @param t_init time stamp of the image where the corner was extracted
     */
    EkltPatch(const cv::Point2d &center, double t_init, const x::Params &params)
      : AsyncPatch()
      , init_center_(center)
      , flow_angle_(0)
      , event_counter_(0)
      , t_init_(t_init)
      , tracking_quality_(1)
      , color_(0, 0, 255)
      , lost_(false)
      , initialized_(false) {
      warping_ = cv::Mat::eye(3, 3, CV_64F);

      patch_size_ = params.eklt_patch_size;
      half_size_ = (params.eklt_patch_size - 1) / 2;
      batch_size_ = static_cast<size_t>(params.eklt_batch_size);
//        update_rate_ = params.eklt_update_every_n_events;
      auto grad = std::make_pair(cv::Mat::zeros(patch_size_, patch_size_, CV_64F),
                                 cv::Mat::zeros(patch_size_, patch_size_, CV_64F));

      reset(init_center_, t_init, grad);
    }

    // TODO find alternative to ros::Time::now().toSec() -- now -1
    explicit EkltPatch(const x::Params &params) : EkltPatch(cv::Point2f(-1, -1), -1, params) {
      // contstructor for initializing lost features
      lost_ = true;
    }

    /**
     * @brief contains checks if event is contained in square around the current feature position
     */
    inline bool contains(double x, double y) const {
      return half_size_ >= std::abs(x - getCenter().x) && half_size_ >= std::abs(y - getCenter().y);
    }

    /**
     * @brief checks if 2x2 ((x,y) to (x+1,y+1)) update is within patch boundaries
     */
    inline bool contains_patch_update(int x, int y) const {
      return (((x + 1) < patch_size_) && (x >= 0) &&
              ((y + 1) < patch_size_) && (y >= 0));
    }

    /**
     * @brief insert an event in the event frame of the patch, incrementing the corresponding pixel by the polarity of the
     * event.
     */
    inline void insert(const x::Event &event) {
      event_buffer_.push_front(event);

      if (event_buffer_.size() > batch_size_) {
        event_buffer_.pop_back();
      }

      event_counter_++;
    }

    /**
     * @brief updates patch center by warping init_center_ with the current warping parameters
     */
    inline void updateCenter(double t) {
      cv::Point2d new_center;
      warpPixel(init_center_, new_center);
      updateTrack(t, new_center);
    }

    /**
     * @brief warpPixel applies the inverse of the linear warp to unwarped and writes to warped.
     */
    inline void warpPixel(cv::Point2d unwarped, cv::Point2d &warped) {
      // compute the position of the feature according to the warp (equation (8) in the paper)
      cv::Mat W = warping_.inv();

      warped.x = W.at<double>(0, 0) * unwarped.x + W.at<double>(0, 1) * unwarped.y + W.at<double>(0, 2);
      warped.y = W.at<double>(1, 0) * unwarped.x + W.at<double>(1, 1) * unwarped.y + W.at<double>(1, 2);
    }

    /**
     * @brief getEventFrame returns the event_frame (according to equation (2) in the paper and resets the counter.
     */
    inline double getEventFrame(cv::Mat &event_frame) {
//        EASY_BLOCK("Event accumulation");
      // implements the function (2) in the paper
      event_frame = cv::Mat::zeros(2 * half_size_ + 1, 2 * half_size_ + 1, CV_64F);

      size_t iterations = batch_size_ < event_buffer_.size() ? batch_size_ - 1 : event_buffer_.size() - 1;
      for (size_t i = 0; i < iterations; ++i) {
        x::Event &e = event_buffer_[i];

        // TODO(JN)
        // heap corruption was occuring at the event_frame.at(y+1,x+1) line below
        // the previous contains function didn't seem to account for +1 accesses:
        // if (!contains(e.x, e.y))
        //    continue;
        // the following solution is temporary and should be revisited
        // e.x is uint16_t, half_size_ is int, center_.x is double
        // what should the rounding policy be from dx to x? is float->int truncation fine?
        double dx = e.x - getCenter().x + half_size_;
        double dy = e.y - getCenter().y + half_size_;
        int x = static_cast<int>(dx);
        int y = static_cast<int>(dy);
        if (!contains_patch_update(x, y)) {
          continue;
        }
        double rx = dx - x;
        double ry = dy - y;

        int increment = e.polarity ? 1 : -1;

        event_frame.at<double>(y + 1, x + 1) += increment * rx * ry;
        event_frame.at<double>(y, x + 1) += increment * rx * (1 - ry);
        event_frame.at<double>(y + 1, x) += increment * (1 - rx) * ry;
        event_frame.at<double>(y, x) += increment * (1 - rx) * (1 - ry);
      }

      // timstamp the image at the middle of the event batch
      event_counter_ = 0;
      return 0.5 * (event_buffer_[0].ts + event_buffer_[iterations].ts);
    }

    /**
     * @brief resets patch after it has been lost. 
     */
    inline void reset(const cv::Point2d &init_center, double t, const std::pair<cv::Mat, cv::Mat>& gradient_xy) {
      // reset feature after it has been lost
      event_counter_ = 0;
      tracking_quality_ = 1;
      lost_ = false;
      initialized_ = false;
      flow_angle_ = 0;

      gradient_xy_ = gradient_xy;

      resetTrack(t, init_center);

      init_center_ = init_center;
      t_init_ = t;

      warping_ = cv::Mat::eye(3, 3, CV_64F);
      event_buffer_.clear();

      // set random color
      static std::uniform_real_distribution<double> unif(0, 255);
      static std::default_random_engine re;
      color_ = cv::Scalar(unif(re), unif(re), unif(re));

      assignNewId();
    }

    cv::Point2d init_center_;
    cv::Mat warping_;
    double flow_angle_;

    int patch_size_;
    int half_size_;

    size_t event_counter_;

    double t_init_;

    double tracking_quality_;
    cv::Scalar color_;

    size_t batch_size_;
//    int update_rate_;
    bool lost_;
    bool initialized_;

    std::deque<x::Event> event_buffer_;

    std::pair<cv::Mat, cv::Mat> gradient_xy_;
  };

}
