//
// Created by frockenb on 06.05.21.
//

#include <x/events/event_accumulator.h>
#include <iostream>

using namespace x;

EventAccumulator::EventAccumulator()
{}

EventAccumulator::EventAccumulator(const size_t buffer_size, const int accumulation_method, const int image_width, const int image_height, std::shared_ptr<Ekf> ekf_pointer)
{
  ekf_pointer_ = ekf_pointer;

  buffer_start_time_ = 0.0;
  buffer_end_time_ = 0.0;
  new_events_ = 0;
  last_used_event_time_ = 0.0;

  event_accumulation_method_ = (AccumulationMethode) accumulation_method;
  event_buffer_.set_capacity(buffer_size);

  time_surface_ = cv::Mat::zeros(image_height, image_width, CV_64FC1);
}

bool EventAccumulator::storeEventsInBuffer(const x::EventArray::ConstPtr &new_events, const Params& params)
{
  for(auto const& e : new_events->events)
  {
    event_buffer_.push_back(e);
    new_events_++;

    if(event_accumulation_method_ == EAM_TIME_SURFACE)
    {
      time_surface_.at<double>(e.y, e.x) = e.ts;
    }
  }
  buffer_start_time_ = event_buffer_.front().ts;
  buffer_end_time_ = event_buffer_.back().ts;

  bool trigger_update = false;

  switch (event_accumulation_method_) {
    case EAM_CONSTANT_ACCUMULATION_TIME:
      if (buffer_end_time_ >= last_used_event_time_ + params.event_accumulation_period) trigger_update = true;
    break;

    case EAM_CONSTANT_FRAME_SIZE:
    default:
        if (new_events_ > params.n_events_min) trigger_update = true;
  }

  if (trigger_update)
  {
    new_events_ = 0;
    last_used_event_time_ = buffer_end_time_;
  }

  return trigger_update;
}

bool EventAccumulator::processEventBuffer(cv::Mat& event_img, double &image_time, const Params& params, const Camera& camera)
{
  // Define temporal window for event accumulation.
  double t0 = event_buffer_.front().ts; // TODO(frockenb) replace with IMU time stamps
  double t1 = event_buffer_.back().ts;

  // Determine the number of events used for noise detection.
  size_t n_events_for_noise_detection = std::min(event_buffer_.size(), size_t(params.n_events_noise_detection_min));

  // Calculate average event rate over temporal window.
  double event_rate = 0.0;
  if (event_buffer_.size() > n_events_for_noise_detection)
  {
    if (0.0 < (event_buffer_.back().ts
               - event_buffer_.at(event_buffer_.size()-n_events_for_noise_detection).ts))
    {
      event_rate = n_events_for_noise_detection /
                   (event_buffer_.back().ts -
                    event_buffer_.at(event_buffer_.size()-n_events_for_noise_detection).ts);
    }
  }

  // Check if event rate is above noise threshold.
  if (event_rate < params.noise_event_rate)
  {
#ifdef DEBUG
    std::cout << "Event rate at timestamp " << event_buffer_.front().ts
              << " below threshold." << event_rate << std::endl;
#endif
    // Return invalid state
    return false;
  }
  else {
    // Calculate index of oldest event used for event accumulation.
    // EDIT (FM): not used, getting rid of compiler warnings
//    int first_event_idx = std::max((int) event_buffer_.size() - params.n_events_max, 0);

    // Calculate time interval in which all events lie.
    // EDIT (FM): not used, getting rid of compiler warnings
//    double temporal_window_size = event_buffer_.back().ts - event_buffer_.at(first_event_idx).ts;

#ifdef DEBUG
    if (event_buffer_.size() < params.n_events_max) {
      std::cout << "Requested frame size of length " << params.n_events_max
                << " events, but I only have " << event_buffer_.size()
                << " events in the last event array." << std::endl;
    }
#endif

    Eigen::Matrix4d T_C_B;
    T_C_B.setIdentity();
    T_C_B.block<3, 3>(0, 0) = params.q_ic.toRotationMatrix();
    T_C_B.block<3, 1>(0,3) = params.p_ic;

    Eigen::Matrix4d T_IMU_1_0;
    T_IMU_1_0.setIdentity();

    auto time_start = ekf_pointer_->getClosestState(t0).getTime();
    auto time_end = ekf_pointer_->getClosestState(t1).getTime();

    double inverse_depth_estimate = params.rho_0;

    if (time_start >= 0 and time_end >= 0) {
      State state_start = ekf_pointer_->getClosestState(t0);
      State state_end = ekf_pointer_->getClosestState(t1);

      if ((time_start != kInvalid) and (time_end != kInvalid)) {
        Eigen::Matrix4d T_start, T_end, T_start_inv;
        T_start.setIdentity();
        T_start.block<3, 3>(0, 0) = state_start.getOrientation().toRotationMatrix();
        T_start.block<3, 1>(0,3) = state_start.getPosition();
        T_end.setIdentity();
        T_end.block<3, 3>(0, 0) = state_end.getOrientation().toRotationMatrix();
        T_end.block<3, 1>(0,3) = state_end.getPosition();
        T_start_inv.setIdentity();
        T_start_inv.block<3, 3>(0, 0) = T_start.block<3, 3>(0, 0).transpose();
        T_start_inv.block<3, 1>(0,3) = T_start.block<3, 3>(0, 0).transpose()* (-T_start.block<3, 1>(0,3));

        T_IMU_1_0 = T_end*T_start_inv;

        if (ekf_pointer_->calculateInverseDepthAverage(time_start, time_end) != 0.0) {
          inverse_depth_estimate = ekf_pointer_->calculateInverseDepthAverage(time_start, time_end);
        }
      }
    }

    Eigen::Matrix4d T_C_B_inv;
    T_C_B_inv.setIdentity();
    T_C_B_inv.block<3, 3>(0, 0) = T_C_B.block<3, 3>(0, 0).transpose();
    T_C_B_inv.block<3, 1>(0,3) = T_C_B.block<3, 3>(0, 0).transpose()* (-T_C_B.block<3, 1>(0,3));

    //Eigen::Matrix4d T_1_0 = T_C_B * T_IMU_1_0 * T_C_B.inverse();
    Eigen::Matrix4d T_1_0 = T_C_B * T_IMU_1_0 * T_C_B_inv;

    drawEventFrame(event_img, T_1_0, time_start, time_end, inverse_depth_estimate, params, camera);
    //drawEventFrame(event_img, T_1_0, t0, t1, inverse_depth_estimate, params, camera);

    image_time = t1;

    return true;
  }
}

void EventAccumulator::drawEventFrame(cv::Mat& event_img, Eigen::Matrix4d T_1_0, double t_start, double t_end, double inv_depth_est, const Params& params, const Camera& camera)
{
  //TODO(frockenb): Figure out what to do with T_1_0, t_start, t_end

  Eigen::Matrix4d K, K_inv;
  K << params.cam_fx*params.img_width,       0.0,                  params.cam_cx*params.img_width,  0.0,
      0.0,                  params.cam_fy*params.img_height,        params.cam_cy*params.img_height,  0.0,
      0.0,                  0.0,                  1.0,            0.0,
      0.0,                  0.0,                  0.0,            1.0;

  K_inv << 1.0/(params.cam_fx*params.img_width), 0.0,                -params.cam_cx/params.cam_fx,   0.0,
      0.0,                    1.0/(params.cam_fy*params.img_height), -params.cam_cy/params.cam_fy,   0.0,
      0.0,                    0.0,                1.0,                            0.0,
      0.0,                    0.0,                0.0,                            1.0;

  //Eigen::Matrix4d T = K * T_1_0 * K_inv; //TODO(frockenb): Check if inverse is calculated correctly  here. // Change Set 1
  Eigen::Matrix4d T = K * T_1_0;

  double dt = 0.0;
  size_t event_counter = 0;

  size_t num_corr = 0;

  //double sum_x = 0.0, sum_y = 0.0;
  //for(auto e = event_buffer_.begin() + first_event_idx; e !=event_buffer_.end(); ++e) //TODO(frockenb): Define how to decide what to accumulate
  for(auto const& e : event_buffer_)
  {
    if ((event_accumulation_method_ == EAM_CONSTANT_ACCUMULATION_TIME) && ((event_buffer_.back().ts - e.ts) > params.event_accumulation_period))
    {
      continue;
    }
    if (params.correct_event_motion)
    {
      if (event_counter % 10 == 0)
      {
        dt = static_cast<float>(t_end - e.ts) / (t_end - t_start);
      }

      Eigen::Vector4d f;
      //f.head<2>() = camera.getKeypoint(e.x + e.y * params.img_width); // Change Set 1
      //f[2] = 1.0; // Change Set 1
      f = camera.getBearing(e.x + e.y * camera.getWidth());
      f.head<3>() /= f[2]; // Change Set 1
      f[3] = inv_depth_est;

      if (dt>0) {
        //f = (1.0 - dt) * f + dt * (T * f); // Change Set 1
        f = (1.0 - dt) * (K * f) + dt * (T * f);
        ++num_corr;
      } else {
        f = K * f;
      }
      cv::Point2d feature;// Change Set 1
      feature.x = f[0]; // Change Set 1
      feature.y = f[1];// Change Set 1
      camera.distort(feature, feature);// Change Set 1

      int x0 = std::floor(feature.x);
      int y0 = std::floor(feature.y);

      //sum_x += abs(e.x - feature.x);
      //sum_y += abs(e.y - feature.y);

      //std::cout << "\nxc: " << x0 << "\nyc: " << y0;

      if (x0 >= 0 && x0 < params.img_width-1 && y0 >= 0 && y0 < params.img_height-1)
      {
        //const float fx = f[0] - x0, fy = f[1] - y0;// Change Set 1
        const float fx = feature.x - x0, fy = feature.y - y0;

        Eigen::Vector4f w((1.f-fx)*(1.f-fy),
                          (fx)*(1.f-fy),
                          (1.f-fx)*(fy),
                          (fx)*(fy));

        event_img.at<float>(y0, x0) += w[0];
        event_img.at<float>(y0, x0+1) += w[1];
        event_img.at<float>(y0+1, x0) += w[2];
        event_img.at<float>(y0+1, x0+1) += w[3];
        //ostrm << e.x << " " << e.y << " " << feature.x << " " << feature.y << " " << dt << " " << T.block<3, 1>(0,3).norm() << std::endl;
      }
      event_counter++;

    } else {
      event_img.at<float>(e.y, e.x) += 1;
    }
  }
}

bool EventAccumulator::renderTimeSurface(cv::Mat& time_surface_img, double &image_time, const Params& params)
{
  cv::exp((time_surface_-buffer_end_time_) / params.event_accumulation_period, time_surface_img);
  //cv::normalize(time_surface_img, time_surface_img, 0, 1, cv::NORM_MINMAX, CV_32FC1);
  image_time = buffer_end_time_;
  return true;
}