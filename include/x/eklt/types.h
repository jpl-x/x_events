#pragma once

#include <deque>

#include <opencv2/core.hpp>
#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>
#include <x/common/event_types.h>

namespace eklt
{

struct Patch; //forward decl
using Patches = std::vector<Patch>; //forward decl

using EventBuffer = std::deque<::x::Event>;
using ImageBuffer = std::map<double, cv::Mat>;

}

namespace eklt
{

struct FeatureTrackData
{
    eklt::Patches patches;
    double t, t_init;
    cv::Mat image;
};

}

namespace eklt
{

using Grid = ceres::Grid2D<double, 2>;
using GridPtr = std::unique_ptr<Grid>;
using Interpolator = ceres::BiCubicInterpolator<Grid>;
using InterpolatorPtr = std::unique_ptr<ceres::BiCubicInterpolator<Grid>>;

struct OptimizerDatum
{
    OptimizerDatum() {}
    OptimizerDatum(const std::vector<double> &grad, const cv::Mat& img, int num_patches)
    {
        grad_ = grad;
        grad_grid_ = new eklt::Grid(grad_.data(), 0, img.rows, 0, img.cols);
        grad_interp_ = new eklt::Interpolator(*grad_grid_);
        ref_counter_ = num_patches;
    }

    void clear()
    {
         delete grad_interp_;
         delete grad_grid_;
    }

    std::vector<double> grad_;
    eklt::Grid* grad_grid_;
    eklt::Interpolator* grad_interp_;
    int ref_counter_;
};

using OptimizerData = std::map<double, OptimizerDatum>;

}



