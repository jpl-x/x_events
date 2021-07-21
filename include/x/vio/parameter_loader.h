//
// Created by Florian Mahlknecht on 2021-03-22.
// Copyright (c) 2021 NASA / JPL. All rights reserved.

#pragma once

#include <x/vio/types.h>
#include <x/eklt/types.h>
#include <x/vio/tools.h>

#include <iostream>

namespace x {

  /**
   * Abstracts functionality of loading
   */
  struct ParameterLoader {

  public:
    /**
     * This is set up as a template function, to allow the usage of polymorphic lambda expressions, making it possible
     * to abstract nh.getParams(key, MULTITYPE) functionality and replace it with ROS master free approaches.
     * @see https://stackoverflow.com/a/51341598
     * @tparam ParamGetFunction
     * @param params_
     * @return
     */
    template<class ParamGetFunction>
    bool loadXParams(Params & params, ParamGetFunction getParam)  {
      // Import initial state or kill xVIO
      bool success = true;
      bool partial_success;

      partial_success = loadInitialStateParams(params, getParam);
      success &= partial_success;

      if (!partial_success)
        std::cout << "ERROR: Initial state parameters are missing!" << std::endl;

      // Import camera calibration or kill xVIO.
      partial_success = loadCameraParams(params, getParam);
      success &= partial_success;

      if (!partial_success)
        std::cout << "ERROR: Camera parameters are missing!" << std::endl;

      partial_success = loadLaserRangeParams(params, getParam);
      success &= partial_success;

      if (!partial_success)
        std::cout << "ERROR: Laser range finder parameters are missing!" << std::endl;

      partial_success = loadSunSensorParams(params, getParam);
      success &= partial_success;

      if (!partial_success)
        std::cout << "ERROR: Sun sensor parameters are missing!" << std::endl;

      partial_success = loadIMUParams(params, getParam);
      success &= partial_success;

      if (!partial_success)
        std::cout << "ERROR: IMU parameters are missing!" << std::endl;

      // Import visual front end parameters or kill xVIO
      partial_success = loadVisualFrontendParams(params, getParam);
      success &= partial_success;

      if (!partial_success)
        std::cout << "ERROR: Visual front end parameters are missing!" << std::endl;

      partial_success = loadFilterParams(params, getParam);
      success &= partial_success;

      if (!partial_success)
        std::cout << "ERROR: Filter parameters are missing!" << std::endl;

      partial_success = loadEventCameraParams(params, getParam);
      success &= partial_success;

      if (!partial_success)
        std::cout << "ERROR: Event camera parameters are missing!" << std::endl;

      partial_success = loadEventAccumulationParams(params, getParam);
      success &= partial_success;

      if (!partial_success)
        std::cout << "ERROR: Event accumulation parameters are missing!" << std::endl;

      partial_success = loadEkltParams(params, getParam);
      success &= partial_success;

      if (!partial_success)
        std::cout << "ERROR: EKLT parameters are missing!" << std::endl;

      partial_success = loadHasteParams(params, getParam);
      success &= partial_success;

      if (!partial_success)
        std::cout << "ERROR: HASTE parameters are missing!" << std::endl;

      return success;
    }

  private:
    template<class ParamGetFunction>
    bool loadCameraParams(Params & params, ParamGetFunction getParam)  {
      std::vector<double> p_ic, q_ic;
      bool success = getParam("cam1_fx", params.cam_fx);
      success &= getParam("cam1_fy", params.cam_fy);
      success &= getParam("cam1_cx", params.cam_cx);
      success &= getParam("cam1_cy", params.cam_cy);
      std::string dist_model;
      success &= getParam("cam1_dist_model", dist_model);
      params.cam_distortion_model = x::stringToDistortionModel(dist_model);
      success &= getParam("cam1_dist_params", params.cam_distortion_parameters);
      success &= getParam("cam1_img_height", params.img_height);
      success &= getParam("cam1_img_width", params.img_width);
      success &= getParam("cam1_p_ic", p_ic);
      success &= getParam("cam1_q_ic", q_ic);
      success &= getParam("cam1_time_offset", params.time_offset);
      success &= getParam("sigma_img", params.sigma_img);
      if (success) {
        // convert to Eigen types
        params.p_ic << p_ic[0], p_ic[1], p_ic[2];
        params.q_ic.w() = q_ic[0];
        params.q_ic.x() = q_ic[1];
        params.q_ic.y() = q_ic[2];
        params.q_ic.z() = q_ic[3];
        params.q_ic.normalize();
      }
      return success;
    }

    template<class ParamGetFunction>
    bool loadInitialStateParams(Params & params, ParamGetFunction getParam) {
      std::vector<double> p, v, b_w, b_a, q, sigma_dp, sigma_dv, sigma_dtheta, sigma_dbw, sigma_dba, g;

      bool success = true;

      success &= getParam("p", p);
      success &= getParam("v", v);
      success &= getParam("q", q);
      success &= getParam("b_w", b_w);
      success &= getParam("b_a", b_a);
      success &= getParam("sigma_dp", sigma_dp);
      success &= getParam("sigma_dv", sigma_dv);
      success &= getParam("sigma_dtheta", sigma_dtheta);
      success &= getParam("sigma_dbw", sigma_dbw);
      success &= getParam("sigma_dba", sigma_dba);
      success &= getParam("g", g);

      if (success) {
        params.p << p[0], p[1], p[2];
        params.v << v[0], v[1], v[2];
        params.q.w() = q[0];
        params.q.x() = q[1];
        params.q.y() = q[2];
        params.q.z() = q[3];
        params.q.normalize();
        params.b_w << b_w[0], b_w[1], b_w[2];
        params.b_a << b_a[0], b_a[1], b_a[2];
        params.g << g[0], g[1], g[2];
        params.sigma_dp << sigma_dp[0], sigma_dp[1], sigma_dp[2];
        params.sigma_dv << sigma_dv[0], sigma_dv[1], sigma_dv[2];
        params.sigma_dtheta << sigma_dtheta[0], sigma_dtheta[1], sigma_dtheta[2];
        params.sigma_dbw << sigma_dbw[0], sigma_dbw[1], sigma_dbw[2];
        params.sigma_dba << sigma_dba[0], sigma_dba[1], sigma_dba[2];
      }

      return success;
    }

    template<class ParamGetFunction>bool loadLaserRangeParams(Params & params, ParamGetFunction getParam) {
      bool success = true;
      // Laser range finder
      success &= getParam("sigma_range", params.sigma_range);
      return success;
    }

    template<class ParamGetFunction>
    bool loadSunSensorParams(Params & params, ParamGetFunction getParam) {
      // Eigen is not supported by ROS' getParam function so we import these
      // variable as std::vectors first
      std::vector<double> q_sc, w_s;

      bool success = true;

      // Import sun sensor calibration or kill xVIO.
      success &= getParam("q_sc", q_sc);
      success &= getParam("w_s", w_s);

      if (success) {
        // Convert std::vectors to msc vectors and quaternions in params
        params.q_sc.w() = q_sc[0];
        params.q_sc.x() = q_sc[1];
        params.q_sc.y() = q_sc[2];
        params.q_sc.z() = q_sc[3];
        params.q_sc.normalize();
        params.w_s << w_s[0], w_s[1], w_s[2];
        params.w_s.normalize();
      }
      return success;
    }

    template<class ParamGetFunction>
    bool loadIMUParams(Params & params, ParamGetFunction getParam) {
      bool success = true;
      success &= getParam("n_a", params.n_a);
      success &= getParam("n_ba", params.n_ba);
      success &= getParam("n_w", params.n_w);
      success &= getParam("n_bw", params.n_bw);
      success &= getParam("a_m_max", params.a_m_max);
      return success;
    }

    template<class ParamGetFunction>
    bool loadVisualFrontendParams(Params & params, ParamGetFunction getParam) {
      bool success = true;
      success &= getParam("fast_detection_delta", params.fast_detection_delta);
      success &= getParam("non_max_supp", params.non_max_supp);
      success &= getParam("block_half_length", params.block_half_length);
      success &= getParam("margin", params.margin);
      success &= getParam("n_feat_min", params.n_feat_min);
      success &= getParam("outlier_method", params.outlier_method);
      success &= getParam("outlier_param1", params.outlier_param1);
      success &= getParam("outlier_param2", params.outlier_param2);
      success &= getParam("n_tiles_h", params.n_tiles_h);
      success &= getParam("n_tiles_w", params.n_tiles_w);
      success &= getParam("max_feat_per_tile", params.max_feat_per_tile);
      return success;
    }

    template<class ParamGetFunction>
    bool loadFilterParams(Params & params, ParamGetFunction getParam) {
      bool success = true;
      // Import filter parameters or kill xVIO
      success &= getParam("n_poses_max", params.n_poses_max);
      success &= getParam("n_slam_features_max", params.n_slam_features_max);
      success &= getParam("rho_0", params.rho_0);
      success &= getParam("sigma_rho_0", params.sigma_rho_0);
      success &= getParam("iekf_iter", params.iekf_iter);
      success &= getParam("msckf_baseline", params.msckf_baseline);
      success &= getParam("min_track_length", params.min_track_length);
      success &= getParam("traj_timeout_gui", params.traj_timeout_gui);
      success &= getParam("init_at_startup", params.init_at_startup);
      success &= getParam("state_buffer_size", params.state_buffer_size);
      success &= getParam("delta_seq_imu", params.delta_seq_imu);
      success &= getParam("state_buffer_time_margin", params.state_buffer_time_margin);
      return success;
    }

    template<class ParamGetFunction>
    bool loadEkltParams(Params & params, ParamGetFunction getParam) {
      bool success = true;
      // Import EKLT parameters
      success &= getParam("eklt_max_corners", params.eklt_max_corners);
      success &= getParam("eklt_min_corners", params.eklt_min_corners);
      success &= getParam("eklt_log_eps", params.eklt_log_eps);
      success &= getParam("eklt_use_linlog_scale", params.eklt_use_linlog_scale);
      success &= getParam("eklt_first_image_t", params.eklt_first_image_t);
      success &= getParam("eklt_lk_window_size", params.eklt_lk_window_size);
      success &= getParam("eklt_num_pyramidal_layers", params.eklt_num_pyramidal_layers);
      success &= getParam("eklt_batch_size", params.eklt_batch_size);
      success &= getParam("eklt_patch_size", params.eklt_patch_size);
      success &= getParam("eklt_max_num_iterations", params.eklt_max_num_iterations);
      success &= getParam("eklt_displacement_px", params.eklt_displacement_px);
      success &= getParam("eklt_tracking_quality", params.eklt_tracking_quality);
      success &= getParam("eklt_bootstrap", params.eklt_bootstrap);
      success &= getParam("eklt_update_every_n_events", params.eklt_update_every_n_events);
      success &= getParam("eklt_scale", params.eklt_scale);
      success &= getParam("eklt_arrow_length", params.eklt_arrow_length);
      success &= getParam("eklt_display_features", params.eklt_display_features);
      success &= getParam("eklt_display_feature_id", params.eklt_display_feature_id);
      success &= getParam("eklt_display_feature_patches", params.eklt_display_feature_patches);

      success &= loadAsyncFrontendParams("eklt_", params.eklt_async_frontend_params, getParam);

      std::string patch_timestamp_assignment;
      success &= getParam("eklt_patch_timestamp_assignment", patch_timestamp_assignment);
      params.eklt_patch_timestamp_assignment = x::stringToEkltPatchTimestampAssignment(patch_timestamp_assignment);

      return success;
    }

    template<class ParamGetFunction>
    bool loadHasteParams(Params & params, ParamGetFunction getParam) {
      bool success = true;
      // Import EKLT parameters
      success &= getParam("haste_patch_size", params.haste_patch_size);
      success &= getParam("haste_max_corners", params.haste_max_corners);
      success &= getParam("haste_min_corners", params.haste_min_corners);

      success &= loadAsyncFrontendParams("haste_", params.haste_async_frontend_params, getParam);

      std::string haste_tracker_type;
      success &= getParam("haste_tracker_type", haste_tracker_type);
      params.haste_tracker_type = x::stringToHasteTrackerType(haste_tracker_type);

      return success;
    }

    template<class ParamGetFunction>
    bool loadAsyncFrontendParams(const std::string& prefix, AsyncFrontendParams & params, ParamGetFunction getParam) {
      bool success = true;

      success &= getParam(prefix + "enable_outlier_removal", params.enable_outlier_removal);
      success &= getParam(prefix + "outlier_method", params.outlier_method);
      success &= getParam(prefix + "outlier_param1", params.outlier_param1);
      success &= getParam(prefix + "outlier_param2", params.outlier_param2);

      std::string ekf_feature_interpolation;
      success &= getParam(prefix + "ekf_feature_interpolation", ekf_feature_interpolation);
      params.ekf_feature_interpolation = x::stringToInterpolationStrategy(ekf_feature_interpolation);
      success &= getParam(prefix + "ekf_feature_extrapolation_limit", params.ekf_feature_extrapolation_limit);
      success &= getParam(prefix + "ekf_update_every_n", params.ekf_update_every_n);

      std::string ekf_update_strategy;
      success &= getParam(prefix + "ekf_update_strategy", ekf_update_strategy);
      params.ekf_update_strategy = x::stringToEkfUpdateStrategy(ekf_update_strategy);
      std::string ekf_update_timestamp;
      success &= getParam(prefix + "ekf_update_timestamp", ekf_update_timestamp);
      params.ekf_update_timestamp = x::stringToEkfUpdateTimestamp(ekf_update_timestamp);

      success &= getParam(prefix + "harris_block_size", params.detection_harris_block_size);
      success &= getParam(prefix + "detection_min_distance", params.detection_min_distance);
      success &= getParam(prefix + "harris_k", params.detection_harris_k);
      success &= getParam(prefix + "harris_quality_level", params.detection_harris_quality_level);

      return success;
    }

    template<class ParamGetFunction>
    bool loadEventCameraParams(Params & params, ParamGetFunction getParam) {
      std::vector<double> p_ec, q_ec;
      bool success = true;
      // Import event parameters or kill xVIO
      /*success &= getParam("event_cam1_fx", params.event_cam_fx);
      success &= getParam("event_cam1_fy", params.event_cam_fy);
      success &= getParam("event_cam1_cx", params.event_cam_cx);
      success &= getParam("event_cam1_cy", params.event_cam_cy);
      std::string event_dist_model;
      success &= getParam("event_cam1_dist_model", event_dist_model);
      params.event_cam_distortion_model = x::stringToDistortionModel(event_dist_model);
      success &= getParam("event_cam1_dist_params", params.event_cam_distortion_parameters);
      success &= getParam("event_cam1_img_height", params.event_img_height);
      success &= getParam("event_cam1_img_width", params.event_img_width);
      success &= getParam("event_cam1_p_ic", p_ec);
      success &= getParam("event_cam1_q_ic", q_ec);
      success &= getParam("event_cam1_time_offset", params.event_cam_time_offset);
      if (success) {
        // Convert std::vectors to msc vectors and quaternions in params
        params.p_ec << p_ec[0], p_ec[1], p_ec[2];
        params.q_ec.w() = q_ec[0];
        params.q_ec.x() = q_ec[1];
        params.q_ec.y() = q_ec[2];
        params.q_ec.z() = q_ec[3];
        params.q_ec.normalize();

      }*/
      return success;
    }

    template<class ParamGetFunction>
    bool loadEventAccumulationParams(Params & params, ParamGetFunction getParam) {
      bool success = true;
      // Import event accumulation parameters or kill xVIO
      success &= getParam("event_accumulation_methode", params.event_accumulation_method);
      success &= getParam("event_accumulation_period", params.event_accumulation_period);
      success &= getParam("n_events_min", params.n_events_min);
      success &= getParam("n_events_max", params.n_events_max);
      success &= getParam("n_events_noise_detection_min", params.n_events_noise_detection_min);
      success &= getParam("noise_event_rate", params.noise_event_rate);
      success &= getParam("event_norm_factor", params.event_norm_factor);
      success &= getParam("correct_event_motion", params.correct_event_motion);

      return success;
    }

  };
}



