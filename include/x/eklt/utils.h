//
// Created by Florian Mahlknecht on 2021-07-10.
// Copyright (c) 2021 NASA / JPL. All rights reserved.



#pragma once

#include <x/vio/types.h>

namespace x {

  void computeLogImgGradients(const cv::Mat &img, cv::Mat &I_x, cv::Mat &I_y, double log_eps, bool use_linlog_scale=false);
}


