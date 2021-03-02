#pragma once

#include <mutex>

#include <opencv2/core/core.hpp>

#include "patch.h"
#include "types.h"


namespace eklt {
/**
 * @brief viewer object that displays the feature tracks as they are computed.
 */
  class Viewer {
  public:
    explicit Viewer(x::EkltParams params = {});

    ~Viewer() = default;

    /**
     * @brief updates the EKLT parameters
     */
    void setParams(const x::EkltParams &params);

    /**
     * @brief main running thread
     */
    void displayTracks();

    /**
     * @brief initializes the tracking data that is used to generate a preview
     */
    void initViewData(double t);

    /**
     * @brief Used to set the tracking data
     */
    void setViewData(eklt::Patches &patches, double t,
                     eklt::ImageBuffer::iterator image_it);

  private:
    /**
     * @brief function that draws on image
     */
    void drawOnImage(FeatureTrackData &data, cv::Mat &view, cv::Mat &image);
    /**
     * EDIT remove to become ROS free
     * @brief helper function to publish image
     */
//    void publishImage(cv::Mat image, ros::Time stamp, std::string encoding, image_transport::Publisher pub);

    x::EkltParams params_;

    FeatureTrackData feature_track_data_;

    cv::Mat feature_track_view_;

    std::mutex data_mutex_;

    bool got_first_image_;
  };

}
