#pragma once

#include <mutex>
#include <random>
#include <deque>

#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <x/eklt/error.h>
#include <x/eklt/types.h>
#include <x/vision/types.h>

#include <easy/profiler.h>
#include <x/vision/camera.h>


namespace x
{

/**
 * @brief The Patch struct which corresponds to an image patch defined by its center (feature position) and
 * dimension (patch size)
 */
struct Patch
{
    /**
   * @brief Patch constructor for patch. Is defined by its center, linear warping matrix, half size ( half the patch
   * side length and the direction of optical flow in its region.
   * @param center Position of the patch defined by the location of the associated corner
   * @param warping Affine 3x3 matrix describing the current transformation taking pixels in the patch to the initial image.
   * @param flow_angle Angle (rad) of the optical flow in the patch around the initial feature
   * @param half_size half size of the patch side length
   * @param t_init time stamp of the image where the corner was extracted
   */
    Patch(const cv::Point2d& center, double t_init, const x::EkltParams& params, x::Camera* cam)
      : init_center_(center)
      , previous_center_(center)
      , flow_angle_(0)
      , event_counter_(0)
      , t_init_(t_init)
      , t_curr_(t_init)
      , t_previous_(t_init)
      , tracking_quality_(1)
      , color_(0, 0, 255)
      , lost_(false)
      , initialized_(false)
      , camera_ptr_(cam)
    {
        warping_ = cv::Mat::eye(3,3,CV_64F);

        patch_size_ = params.patch_size;
        half_size_ = (params.patch_size - 1) / 2;
        batch_size_ = params.batch_size;
        update_rate_ = params.update_every_n_events;
        ekf_feature_interpolation_ = params.ekf_feature_interpolation;
        ekf_feature_extrapolation_limit_ = params.ekf_feature_extrapolation_limit;

        reset(init_center_, t_init);
    }

    // TODO find alternative to ros::Time::now().toSec() -- now -1
    explicit Patch(const x::EkltParams& params, x::Camera* cam) : Patch(cv::Point2f(-1,-1), -1, params, cam)
    {
        // contstructor for initializing lost features
        lost_ = true;
    }

    Match consumeMatch(double t) {
      Match m;
//      // for now assume no distortion
//      m.previous = Feature(t_previous_, 0, previous_center_.x, previous_center_.y, previous_center_.x, previous_center_.y);
//      m.current = Feature(t_curr_, 0, center_.x, center_.y, center_.x, center_.y);
//      saveCurrentInPrevious();
      m.previous = previous_feature_;
      m.current = interpolateToTime(t);

      camera_ptr_->undistortFeature(m.current);

      previous_feature_ = m.current;

      return m;
    }

    Feature interpolateToTime(double t) const {
      double x, y;

      switch (ekf_feature_interpolation_) {
        case EkltEkfFeatureInterpolation::NEAREST_NEIGHBOR:
          if (fabs(t - t_curr_) < fabs(t - t_previous_)) {
            x = center_.x;
            y = center_.y;
          } else {
            x = previous_center_.x;
            y = previous_center_.y;
          }
          break;
        case EkltEkfFeatureInterpolation::LINEAR_NO_LIMIT: {
          // time factor assembles normed time distance from current center: e.g. -1 --> previous_center
          double time_factor = 0;

          if (fabs(t_curr_ - t_previous_) >= 1e-9)  // avoid division by zero
            time_factor = (t - t_curr_) / (t_curr_ - t_previous_);

          x = center_.x + time_factor * (center_.x - previous_center_.x);
          y = center_.y + time_factor * (center_.y - previous_center_.y);

          break;
        }
        case EkltEkfFeatureInterpolation::LINEAR_RELATIVE_LIMIT: {
          // time factor assembles normed time distance from current center: e.g. -1 --> previous_center
          double time_factor = 0;

          if (fabs(t_curr_ - t_previous_) >= 1e-9)  // avoid division by zero
            time_factor = (t - t_curr_) / (t_curr_ - t_previous_);

          if (time_factor > 0) {
            time_factor = fmin(time_factor, ekf_feature_extrapolation_limit_);
          } else {
            time_factor = fmax(time_factor, -1 - ekf_feature_extrapolation_limit_);
          }

          x = center_.x + time_factor * (center_.x - previous_center_.x);
          y = center_.y + time_factor * (center_.y - previous_center_.y);
          break;
        }
        case EkltEkfFeatureInterpolation::LINEAR_ABSOLUTE_LIMIT: {
          // time factor assembles normed time distance from current center: e.g. -1 --> previous_center
          double time_factor = 0;

          if (fabs(t_curr_ - t_previous_) >= 1e-9) {  // avoid division by zero
            time_factor = (t - t_curr_) / (t_curr_ - t_previous_);

            if (time_factor > 0) {
              time_factor = fmin(time_factor, ekf_feature_extrapolation_limit_ / (t_curr_ - t_previous_));
            } else {
              time_factor = fmax(time_factor, -1 - ekf_feature_extrapolation_limit_ / (t_curr_ - t_previous_));
            }
          }

          x = center_.x + time_factor * (center_.x - previous_center_.x);
          y = center_.y + time_factor * (center_.y - previous_center_.y);
          break;
        }

        case EkltEkfFeatureInterpolation::NO_INTERPOLATION:
        default:
          x = center_.x;
          y = center_.y;
          break;
      }
      return {t, 0, x, y, x, y};
    }

    /**
   * @brief contains checks if event is contained in square around the current feature position
   */
    inline bool contains(double x, double y) const
    {
        return half_size_ >= std::abs(x - center_.x) && half_size_ >= std::abs(y - center_.y);
    }

    /**
   * @brief checks if 2x2 ((x,y) to (x+1,y+1)) update is within patch boundaries
   */
    inline bool contains_patch_update(int x, int y) const
    {
      return (((x+1) < patch_size_) && (x >= 0) &&
              ((y+1) < patch_size_) && (y >= 0));
    }

    /**
   * @brief insert an event in the event frame of the patch, incrementing the corresponding pixel by the polarity of the
   * event.
   */
    inline void insert(const x::Event& event)
    {
        event_buffer_.push_front(event);

        if (event_buffer_.size() > batch_size_)
        {
            event_buffer_.pop_back();
        }

        event_counter_++;
    }

    /**
     * @brief updates patch center by warping init_center_ with the current warping parameters
     */
    inline void updateCenter(double t) {
      // save previous before update
      saveCurrentInPrevious();
      t_curr_ = t;
      warpPixel(init_center_, center_);
    }

  void saveCurrentInPrevious() {
    previous_center_ = center_;
    t_previous_ = t_curr_;
  }

  /**
 * @brief warpPixel applies the inverse of the linear warp to unwarped and writes to warped.
 */
    inline void warpPixel(cv::Point2d unwarped, cv::Point2d &warped)
    {
        // compute the position of the feature according to the warp (equation (8) in the paper)
        cv::Mat W = warping_.inv();

        warped.x = W.at<double>(0,0) * unwarped.x + W.at<double>(0,1) * unwarped.y + W.at<double>(0,2);
        warped.y = W.at<double>(1,0) * unwarped.x + W.at<double>(1,1) * unwarped.y + W.at<double>(1,2);
    }

    /**
   * @brief getEventFramesAndReset returns the event_frame (according to equation (2) in the paper and resets the counter.
   */
    inline void getEventFramesAndReset(cv::Mat &event_frame)
    {
//        EASY_BLOCK("Event accumulation");
        // implements the function (2) in the paper
        event_frame = cv::Mat::zeros(2*half_size_+1, 2*half_size_+1, CV_64F);

        int iterations =  batch_size_< event_buffer_.size() ? batch_size_-1 : event_buffer_.size()-1;
        for (int i=0; i<iterations; i++)
        {
            x::Event &e = event_buffer_[i];

            // TODO(JN)
            // heap corruption was occuring at the event_frame.at(y+1,x+1) line below
            // the previous contains function didn't seem to account for +1 accesses:
            // if (!contains(e.x, e.y))
            //    continue;
            // the following solution is temporary and should be revisited
            // e.x is uint16_t, half_size_ is int, center_.x is double
            // what should the rounding policy be from dx to x? is float->int truncation fine?
            double dx = e.x - center_.x + half_size_;
            double dy = e.y - center_.y + half_size_;
            int x = static_cast<int>(dx);
            int y = static_cast<int>(dy);
            if (!contains_patch_update(x, y)) {
              continue;
            }
            double rx = dx - x;
            double ry = dy - y;

            int increment = e.polarity ? 1 : -1;

            event_frame.at<double>(y+1, x+1) += increment * rx * ry;
            event_frame.at<double>(y  , x+1) += increment * rx * (1 - ry);
            event_frame.at<double>(y+1, x  ) += increment * (1 - rx) * ry;
            event_frame.at<double>(y  , x  ) += increment * (1 - rx) * (1 - ry);
        }

        // timstamp the image at the middle of the event batch
        t_curr_ = 0.5*(event_buffer_[0].ts + event_buffer_[iterations].ts);
        event_counter_ = 0;
    }

    /**
     * @brief resets patch after it has been lost. 
     */
    inline void reset(const cv::Point2d& init_center, double t)
    {
        // reset feature after it has been lost
        event_counter_ = 0;
        tracking_quality_=1;
        lost_ = false;
        initialized_ = false;
        flow_angle_ = 0;
        
        center_ = init_center;
        init_center_ = init_center;
        previous_center_ = init_center;
        t_curr_ = t;
        t_init_ = t;
        t_previous_ = t;
        previous_feature_ = Feature(t_curr_, 0, center_.x, center_.y, center_.x, center_.y);


        warping_ = cv::Mat::eye(3, 3, CV_64F);
        event_buffer_.clear();

        // set random color
        static std::uniform_real_distribution<double> unif(0, 255);
        static std::default_random_engine re;
        color_ = cv::Scalar(unif(re), unif(re), unif(re));

        static int id = 0;
        id_ = id++;
    }

    cv::Point2d init_center_;
    cv::Point2d center_;
    cv::Point2d previous_center_;
    Feature previous_feature_;
    cv::Mat warping_;
    double flow_angle_;

    int patch_size_;
    int half_size_;

    int event_counter_;

    double t_init_;
    double t_curr_;
    double t_previous_;

    double tracking_quality_;
    cv::Scalar color_;

    int id_ {-1};
    int batch_size_;
    int update_rate_;
    bool lost_;
    bool initialized_;
    x::Camera* camera_ptr_;

    std::deque<x::Event> event_buffer_;
    EkltEkfFeatureInterpolation ekf_feature_interpolation_;
    double ekf_feature_extrapolation_limit_;
};

}
