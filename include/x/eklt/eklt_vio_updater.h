//
// Created by Florian Mahlknecht on 2021-03-01.
// Copyright (c) 2021 NASA / JPL. All rights reserved.
//

#pragma once

#include <x/ekf/updater.h>

namespace x {

class EkltVioUpdater : public Updater {

  void preProcess(const State& state) override;
  bool preUpdate(State& state) override;
  void constructUpdate(const State& state, Matrix& h, Matrix& res, Matrix& r) override;
  void postUpdate(State& state, const Matrix& correction) override;
};


}