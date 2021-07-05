//
// Created by Florian Mahlknecht on 2021-07-05.
// Copyright (c) 2021 NASA / JPL. All rights reserved.


#pragma once


// forward declaration instead of include (!)
// #include <x/haste/tracking/hypothesis_tracker.hpp>
namespace haste {
  class HypothesisPatchTracker;
}


// declares HASTE types used within x:: namespace

namespace x {

  using HypothesisTrackerPtr = std::shared_ptr<haste::HypothesisPatchTracker>;

}