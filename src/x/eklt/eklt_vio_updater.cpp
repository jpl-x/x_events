//
// Created by Florian Mahlknecht on 2021-03-01.
// Copyright (c) 2021 NASA / JPL. All rights reserved.
//

#include <x/eklt/eklt_vio_updater.h>

using namespace x;

void EkltVioUpdater::preProcess(const State &state) {

}

bool EkltVioUpdater::preUpdate(State &state) {
  return false;
}

void EkltVioUpdater::constructUpdate(const State &state, Matrix &h, Matrix &res, Matrix &r) {

}

void EkltVioUpdater::postUpdate(State &state, const Matrix &correction) {

}
