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
#include <x/vision/camera_models.h>

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

  if (distortion_model == DistortionModel::FOV) {
    // ZE undistortion code needs this term at index 1
    double tan_s_half_x2 = 2.0 * std::tan(distortion_parameters_[0] / 2.0);
    distortion_parameters_.push_back(tan_s_half_x2);
  }

  calculateBearingLUT();
  calculateKeypointLUT();
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
//  EASY_FUNCTION();

  // change to centered normalized pixel coordinates
  double px[2];
  px[0] = input.x * inv_fx_ - cx_n_;
  px[1] = input.y * inv_fy_ - cy_n_;

  switch (distortion_model_) {
    case DistortionModel::FOV:
      ze::FovDistortion::undistort(distortion_parameters_.data(), px);
      break;
    case DistortionModel::RADIAL_TANGENTIAL:
      ze::RadialTangentialDistortion::undistort(distortion_parameters_.data(), px);
      break;
    case DistortionModel::EQUIDISTANT:
      ze::EquidistantDistortion::undistort(distortion_parameters_.data(), px);
      break;
  }

  // change back to actual pixel values
  undistorted_output.x = px[0] * fx_ + cx_;
  undistorted_output.y = px[1] * fy_ + cy_;
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
  normalized_feature.setYDist(feature.getYDist() * inv_fy_ - cy_n_);

  return normalized_feature;
}
// TODO(frockenb) Port descriptions to header file

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

/** \brief Projects point using basic pinhole camera model.
 *  \param px Point coordinates
 */
void Camera::pinholeProject(Eigen::Vector2d& px) const
{
  px[0] = px[0] * fx_ + cx_;
  px[1] = px[1] * fy_ + cy_;
}

/** \brief Projects point using basic pinhole camera model.
 *  \param px Point coordinates
 */
void Camera::pinholeProject(cv::Point2d& px) const
{
  px.x = px.x * fx_ + cx_;
  px.y = px.y * fy_ + cy_;
}

/** \brief Back projects point using basic pinhole camera model.
 *  \param px Point coordinates
 */
void Camera::pinholeBackProject(Eigen::Vector2d& px) const
{
  px[0] = (px[0] - cx_) * inv_fx_;
  px[1] = (px[1] - cy_) * inv_fy_;
}

/** \brief Back projects point using basic pinhole camera model.
 *  \param px Point coordinates
 */
void Camera::pinholeBackProject(cv::Point2d& px) const
{
  px.x = (px.x - cx_) * inv_fx_;
  px.y = (px.y - cy_) * inv_fy_;
}

/** \brief Back projects bearing vector using basic pinhole camera model.
 *  \param bearing Bearing vector
 */
void Camera::pinholeBackProject(Eigen::Vector3d& bearing) const
{
  bearing[0] = (bearing[0] - cx_) * inv_fx_;
  bearing[1] = (bearing[1] - cy_) * inv_fy_;
}

/** \brief Unit coordinates -> distortion -> pinhole, offset and scale. // TODO(frockenb) write proper description.
 *  \param bearing_out Bearing vector
 *  \return px
 */
cv::Point2d Camera::project(const Eigen::Vector3d& bearing) const
{
  cv::Point2d px;
  px.x = bearing[0] / bearing[2];
  px.y = bearing[1] / bearing[2];

  Camera::distort(px, px);
  Camera::pinholeProject(px);
  return px;
}

/** \brief  // TODO(frockenb) write proper description.
 *  \param keypoint
 *  \return bearing_out Bearing vector
 */
Eigen::Vector3d Camera::backProject(const cv::Point2d& keypoint) const
{
  cv::Point2d keypoint_out = keypoint;

  //Camera::pinholeBackProject(keypoint_out);
  Camera::undistort(keypoint_out, keypoint_out);

  Eigen::Vector3d bearing_out;
  bearing_out << (keypoint_out.x - cx_) * inv_fx_, (keypoint_out.y - cy_) * inv_fy_, 1.0;

  return bearing_out.normalized();
}

/** \brief Precomputes a Look Up Table to convert image coordinates to bearing vectors.
 *  \pre Image and distortion parameters have to be set prior to bearing_lut_ calculation.
 */
void Camera::calculateBearingLUT()
{
  size_t n = img_height_ * img_width_;
  bearing_lut_.resize(4, n);

  cv::Point2d keypoint;

  for (size_t y=0; y != img_height_; ++y)
  {
    for (size_t x=0; x != img_width_; ++x)
    {
      keypoint.x = (double) x;
      keypoint.y = (double) y;
      Eigen::Vector3d f = backProject(keypoint);
      bearing_lut_.col(x + y * img_width_) =
        Eigen::Vector4d(f[0], f[1], f[2], 1.0);
    }
  }
}

/** \brief Precomputes a Look Up Table to convert bearing vectors to image coordinates.
 *  \pre Image and distortion parameters have to be set prior to keypoint_lut_ calculation.
 *  \pre bearing_lit_ has to be computed prior to keypoint_lut_ calculation.
 */
void Camera::calculateKeypointLUT()
{
  size_t n = img_height_ * img_width_;
  keypoint_lut_.resize(2, n);

  for (size_t i=0; i != n; ++i)
  {
    cv::Point2d p = project(bearing_lut_.col(i).head<3>().cast<double>());

    Eigen::Vector2d keypoint;
    keypoint[0] = p.x;
    keypoint[1] = p.y;

    keypoint_lut_.col(i) = keypoint;
  }
}

void Camera::distort(const cv::Point2d &input, cv::Point2d &distorted_output) const
{
  // change to centered normalized pixel coordinates
  double px[2];
  px[0] = input.x * inv_fx_ - cx_n_;
  px[1] = input.y * inv_fy_ - cy_n_;

  switch (distortion_model_) {

    case DistortionModel::FOV:
      ze::FovDistortion::distort(distortion_parameters_.data(), px);
      break;
    case DistortionModel::RADIAL_TANGENTIAL:
      ze::RadialTangentialDistortion::distort(distortion_parameters_.data(), px);
      break;
    case DistortionModel::EQUIDISTANT:
      ze::EquidistantDistortion::distort(distortion_parameters_.data(), px);
      break;
  }

  // change back to actual pixel values
  distorted_output.x = px[0] * fx_ + cx_;
  distorted_output.y = px[1] * fy_ + cy_;
}
