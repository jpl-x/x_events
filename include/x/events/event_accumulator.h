//
// Created by frockenb on 06.05.21.
//

#ifndef X_EVENT_ACCUMULATOR_H
#define X_EVENT_ACCUMULATOR_H

#include <x/vio/types.h>
#include <x/common/event_types.h>
#include <x/vision/camera.h>

namespace x
{
class EventAccumulator
{
public:
  EventAccumulator();

  EventAccumulator(const size_t buffer_size, const int accumulation_method);

  bool storeEventsInBuffer(const x::EventArray::ConstPtr &new_events, const Params& params);

  bool processEventBuffer(cv::Mat& event_img, double &image_time, const Params& params, const Camera& camera);

private:

  double buffer_start_time_;
  double buffer_end_time_;
  EventBuffer event_buffer_;

  enum AccumulationMethode {
    EAM_CONSTANT_ACCUMULATION_TIME = 0,
    EAM_CONSTANT_FRAME_SIZE = 1
  } event_accumulation_method_;
  int new_events_;
  double last_used_event_time_;

  void drawEventFrame(cv::Mat& event_img, Eigen::Matrix4d T_1_0, double t_start, double t_end, const Params& params, const Camera& camera);

};
}

#endif //X_EVENT_ACCUMULATOR_H
