//
// Created by Florian Mahlknecht on 2021-04-03.
// Copyright (c) 2021 NASA / JPL. All rights reserved.

#include <x/eklt/types.h>


using namespace x;

std::ostream& x::operator << (std::ostream& os, const x::EkltTrackUpdateType& obj)
{
  switch (obj) {
    case x::EkltTrackUpdateType::Init:
      os << "Init";
      break;
    case x::EkltTrackUpdateType::Bootstrap:
      os << "Bootstrap";
      break;
    case x::EkltTrackUpdateType::Update:
      os << "Update";
      break;
    case x::EkltTrackUpdateType::Lost:
      os << "Lost";
      break;
    default:
      os << static_cast<uint8_t>(obj);
  }
  return os;
}