//
// Created by Florian Mahlknecht on 2021-03-01.
// Copyright (c) 2021 NASA / JPL. All rights reserved.
//

#pragma once

#include <x/vio/types.h>
#include <x/eklt/types.h>
#include <x/eklt/eklt_tracker.h>
#include <x/eklt/viewer.h>
#include <x/eklt/eklt_vio_updater.h>
#include <x/vio/state_manager.h>
#include <x/vio/track_manager.h>
#include <x/ekf/ekf.h>
#include <x/vision/camera.h>
#include <x/vision/tiled_image.h>
#include <x/vision/tracker.h>

#include <x/common/event_types.h>
#include <x/vio/abstract_vio.h>


namespace x {
  class EKLTVIO : public AbstractVio {
  public:
    explicit EKLTVIO(XVioPerformanceLoggerPtr  xvio_perf_logger = nullptr,
                     EkltPerformanceLoggerPtr  eklt_perf_logger = nullptr);

    bool isInitialized() const;

    void setUp(const Params &params) override;

    void setLastRangeMeasurement(RangeMeasurement range_measurement);

    void setLastSunAngleMeasurement(SunAngleMeasurement angle_measurement);

    void initAtTime(double now) override;

    void getMsckfFeatures(Vector3dArray &inliers, Vector3dArray &outliers);

    /**
     * Pass IMU measurements to EKF for propagation.
     *
     * @param[in] timestamp
     * @param[in] msg_seq Message ID.
     * @param[in] w_m Angular velocity (gyroscope).
     * @param[in] a_m Specific force (accelerometer).
     * @return The propagated state.
     */
    State processImu(double timestamp,
                     unsigned int seq,
                     const Vector3 &w_m,
                     const Vector3 &a_m) override;

    /**
     * Creates an update measurement from image and pass it to EKF.
     *
     * @param[in] timestamp Image timestamp.
     * @param[in] seq Image sequence ID.
     * @param[in,out] match_img Image input, overwritten as tracker debug image
     *                          in output.
     * @param[out] feature_img Track manager image output.
     * @return The updated state, or invalide if the update could not happen.
     */
    State processImageMeasurement(double timestamp,
                                  unsigned int seq,
                                  TiledImage &match_img,
                                  TiledImage &feature_img) override;

    /**
     * Processes events information.
     *
     * @param[in] events_ptr Pointer to event array.
     * @return The updated state, or invalid if the update could not happen.
     */
    State processEventsMeasurement(const x::EventArray::ConstPtr &events_ptr,
                                   TiledImage &tracker_img, TiledImage &feature_img) override;

    bool doesProcessEvents() const override { return true; }

    /**
     * Compute cartesian coordinates of SLAM features for input state.
     *
     * @param[in] state Input state.
     * @return A vector with the 3D cartesian coordinates.
     */
    std::vector<Vector3>
    computeSLAMCartesianFeaturesForState(const State &state);

  private:
    /**
     * Extended Kalman filter estimation back end.
     */
    Ekf ekf_;

    /**
     * VIO EKF updater.
     *
     * Constructs and applies an EKF update from a VIO measurement. The EKF
     * class owns a reference to this object through Updater abstract class,
     * which it calls to apply the update.
     */
    EkltVioUpdater vio_updater_;

    Params params_;

    /**
     * Minimum baseline for MSCKF (in normalized plane).
     */
    double msckf_baseline_n_;

    Camera camera_;
    Tracker tracker_;
    Viewer eklt_viewer_;
    EkltTracker eklt_tracker_;
    TrackManager track_manager_;
    StateManager state_manager_;
    RangeMeasurement last_range_measurement_;
    SunAngleMeasurement last_angle_measurement_;
    bool initialized_{false};

    // counts the asynchronous state updates triggered by the EKLT tracker
    int seq_ = 0;

    // optional performance loggers
    XVioPerformanceLoggerPtr xvio_perf_logger_;
    EkltPerformanceLoggerPtr eklt_perf_logger_;
  };
} // namespace x

