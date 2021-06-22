//
// Created by Florian Mahlknecht on 2021-06-21.
// Copyright (c) 2021 NASA / JPL. All rights reserved.



#pragma once

#include <x/vio/types.h>
#include <x/ekf/state.h>
#include <x/common/event_types.h>

namespace x {

  /**
   * Defines common interface for all VIO classes, to have an easier time evaluating them with a single function
   */

  class AbstractVio {
  public:
    virtual ~AbstractVio() = default;
    virtual void setUp(const x::Params& params) = 0;
    virtual void initAtTime(double now) = 0;
    virtual bool isInitialized() const = 0;

    virtual State processImu(double timestamp, unsigned int seq, const Vector3& w_m, const Vector3& a_m) = 0;
    virtual State processImageMeasurement(double timestamp, unsigned int seq,
                                          TiledImage &match_img, TiledImage &feature_img) = 0;

    virtual bool doesProcessEvents() const { return false; }

    // default empty implementation
    virtual State processEventsMeasurement(const x::EventArray::ConstPtr &events_ptr,
                                           TiledImage &tracker_img, TiledImage &feature_img) { return x::State(); };
  };

}