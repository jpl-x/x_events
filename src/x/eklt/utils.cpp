//
// Created by Florian Mahlknecht on 2021-07-10.
// Copyright (c) 2021 NASA / JPL. All rights reserved.



#include <x/eklt/utils.h>
#include <opencv2/imgproc.hpp>

void x::computeLogImgGradients(const cv::Mat &img, cv::Mat &I_x, cv::Mat &I_y, double log_eps, bool use_linlog_scale) {

  cv::Mat log_image;

  if (use_linlog_scale) {
    const uint8_t threshold = 20;

    cv::Mat mask = img > threshold;

    cv::Mat lin_image, img_float;
    img.convertTo(img_float, CV_64F, 1.0);

    log_image = cv::Mat::ones(img.size(), CV_64F) * threshold;
    img_float.copyTo(log_image, mask);
    cv::log(log_image, log_image);
    img.convertTo(lin_image, CV_64F, 1.0 / threshold * log(threshold));
    lin_image.copyTo(log_image, ~mask);
  } else {
    cv::Mat normalized_image;
    img.convertTo(normalized_image, CV_64F, 1.0 / 255.0);
    cv::log(normalized_image + log_eps, log_image);
  }

  cv::Sobel(log_image / 8, I_x, CV_64F, 1, 0, 3);
  cv::Sobel(log_image / 8, I_y, CV_64F, 0, 1, 3);

}
