/*
 * Copyright 2020 California  Institute  of Technology (“Caltech”)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <x/vision/camera.h>

using namespace x;

Camera::Camera()
{}

Camera::Camera(double fx,
               double fy,
               double cx,
               double cy,
               DistortionModel distortion_model,
               std::vector<double> distortion_parameters,
               unsigned int img_width,
               unsigned int img_height)
: distortion_model_(distortion_model)
, distortion_parameters_(std::move(distortion_parameters))
, img_width_(img_width)
, img_height_(img_height)
{
  fx_ = img_width * fx;
  fy_ = img_height * fy;
  cx_ = img_width * cx;
  cy_ = img_height * cy;

  inv_fx_ = 1.0 / fx_;
  inv_fy_ = 1.0 / fy_;
  cx_n_ = cx_ * inv_fx_;
  cy_n_ = cy_ * inv_fy_;
}

unsigned int Camera::getWidth() const
{
  return img_width_;
}

unsigned int Camera::getHeight() const
{
  return img_height_;
}

double Camera::getInvFx() const
{
  return inv_fx_;
}

double Camera::getInvFy() const
{
  return inv_fy_;
}

double Camera::getCxN() const
{
  return cx_n_;
}

double Camera::getCyN() const
{
  return cy_n_;
}

void Camera::undistort(const cv::Point2d &input, cv::Point2d &undistorted_output) const {
  const double cam_dist_x = input.x * inv_fx_ - cx_n_;
  const double cam_dist_y = input.y * inv_fy_ - cy_n_;

  const double dist_r = sqrt(cam_dist_x * cam_dist_x + cam_dist_y * cam_dist_y);

  double xn;
  double yn;

  switch(distortion_model_) {
    case DistortionModel::FOV: {
      const double& s = distortion_parameters_[0];
      const double s_term = 1.0 / ( 2.0 * std::tan(s / 2.0) );

      double distortion_factor = 1.0;
      if(dist_r > 0.01 && s!= 0.0) {
        distortion_factor = std::tan(dist_r * s) * s_term / dist_r;
      }
      xn = distortion_factor * cam_dist_x;
      yn = distortion_factor * cam_dist_y;

      break;
    }
    case DistortionModel::RADIAL_TANGENTIAL: {
      // @see https://github.com/ethz-asl/image_undistort/blob/master/src/undistorter.cpp
      // Split out parameters for easier reading
      const double& k1 = distortion_parameters_[0];
      const double& k2 = distortion_parameters_[1];
      const double& k3 = distortion_parameters_[4];
      const double& p1 = distortion_parameters_[2];
      const double& p2 = distortion_parameters_[3];

      // Undistort
      const double r2 = cam_dist_x * cam_dist_x + cam_dist_y * cam_dist_y;
      const double r4 = r2 * r2;
      const double r6 = r4 * r2;
      const double kr = (1.0 + k1 * r2 + k2 * r4 + k3 * r6);
      xn = cam_dist_x * kr + 2.0 * p1 * cam_dist_x * cam_dist_y + p2 * (r2 + 2.0 * cam_dist_x * cam_dist_x);
      yn = cam_dist_y * kr + 2.0 * p2 * cam_dist_x * cam_dist_y + p1 * (r2 + 2.0 * cam_dist_y * cam_dist_y);
      break;
    }
    default: {
      std::ostringstream message;
      message << "Distortion model not implemented - model: " << static_cast<int>(distortion_model_);
      throw std::runtime_error(message.str());
    }
  }

  undistorted_output.x = xn * fx_ + cx_;
  undistorted_output.y = yn * fy_ + cy_;
}

void Camera::undistortFeatures(FeatureList& features) const
{
  // Undistort each point in the input vector
  for(auto & feature : features)
    undistortFeature(feature);
}

void Camera::undistortFeature(Feature& feature) const
{
  // this will be simplified as soon as we start saving x, y as cv::Point2d in feature
  cv::Point2d input, output;
  input.x = feature.getXDist();
  input.y = feature.getYDist();
  undistort(input, output);
  feature.setX(output.x);
  feature.setY(output.y);
}

Feature Camera::normalize(const Feature& feature) const
{
  Feature normalized_feature(feature.getTimestamp(),
      feature.getX() * inv_fx_ - cx_n_,
      feature.getY() * inv_fy_ - cy_n_);
  normalized_feature.setXDist(feature.getXDist() * inv_fx_ - cx_n_);
  normalized_feature.setYDist(feature.getYDist() * inv_fx_ - cx_n_);

  return normalized_feature;
}

/** \brief Normalized the image coordinates for all features in the input track
 *  \param track Track to normalized
 *  \param max_size Max size of output track, cropping from the end (default 0: no cropping)
 *  \return Normalized track
 */
Track Camera::normalize(const Track& track, const size_t max_size) const
{
  // Determine output track size
  const size_t track_size = track.size();
  size_t size_out;
  if(max_size)
    size_out = std::min(max_size, track_size);
  else
    size_out = track_size;

  Track normalized_track(size_out, Feature());
  const size_t start_idx(track_size - size_out);
  for (size_t j = start_idx; j < track_size; ++j)
    normalized_track[j - start_idx] = normalize(track[j]);
  
  return normalized_track;
}

/** \brief Normalized the image coordinates for all features in the input track list
 *  \param tracks List of tracks
 *  \param max_size Max size of output tracks, cropping from the end (default 0: no cropping)
 *  \return Normalized list of tracks
 */
TrackList Camera::normalize(const TrackList& tracks, const size_t max_size) const
{
  const size_t n = tracks.size();
  TrackList normalized_tracks(n, Track());
  
  for (size_t i = 0; i < n; ++i)
    normalized_tracks[i] = normalize(tracks[i], max_size);

  return normalized_tracks;
}
