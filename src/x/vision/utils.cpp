//
// Created by Florian Mahlknecht on 2021-04-26.
// Copyright (c) 2021 NASA / JPL. All rights reserved.


#include <x/vision/utils.h>

#include "opencv2/calib3d/calib3d.hpp"


std::vector<uchar> x::detectOutliers(const std::vector<cv::Point2f> &pts1, const std::vector<cv::Point2f> &pts2,
                                     int outlier_method_, double outlier_param1_, double outlier_param2_) {
  std::vector<uchar> mask;
  cv::findFundamentalMat(pts1,
                         pts2,
                         outlier_method_,
                         outlier_param1_,
                         outlier_param2_,
                         mask);
  return mask;
}