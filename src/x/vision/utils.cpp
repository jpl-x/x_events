//
// Created by Florian Mahlknecht on 2021-04-26.
// Copyright (c) 2021 NASA / JPL. All rights reserved.


#include <x/vision/utils.h>
#include <map>

#include "opencv2/calib3d/calib3d.hpp"


std::vector<uchar> x::detectOutliers(const std::vector<cv::Point2f> &pts1, const std::vector<cv::Point2f> &pts2,
                                     int outlier_method_, double outlier_param1_, double outlier_param2_) {

  EASY_BLOCK("RANSAC outlier removal", profiler::colors::Red);
  std::vector<uchar> mask;
  cv::findFundamentalMat(pts1,
                         pts2,
                         outlier_method_,
                         outlier_param1_,
                         outlier_param2_,
                         mask);
  return mask;
}

void x::dumpFrame(x::XVioPerformanceLoggerPtr& csv_logger, double t, const std::string &type, const cv::Mat& frame) {
  if (frame.empty())
    return;

  static std::map<std::string, uint64_t> frame_counts;

  if (frame_counts.find(type) == frame_counts.end()) {
    frame_counts[type] = 0;
  }

  std::ostringstream ss;
  ss << type << "_" << std::setw(6) << std::setfill('0') << frame_counts[type]++ << ".png";
  std::string filename(ss.str());

  std::string file = csv_logger->frames_path / filename;

  cv::imwrite(file, frame);

  csv_logger->dumped_frames_csv.addRow(profiler::now(), t, type, filename);
}
