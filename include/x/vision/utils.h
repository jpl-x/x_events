//
// Created by Florian Mahlknecht on 2021-04-26.
// Copyright (c) 2021 NASA / JPL. All rights reserved.



#pragma once

#include <x/vision/types.h>


namespace x {
  std::vector<uchar> detectOutliers(const std::vector<cv::Point2f> &pts1, const std::vector<cv::Point2f> &pts2,
                                    int outlier_method_ = 8, double outlier_param1_ = 0.3, double outlier_param2_ = 0.99);
}


