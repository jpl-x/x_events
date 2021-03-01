//
// Created by Florian Mahlknecht on 2021-02-19.
// Copyright (c) 2021 NASA / JPL. All rights reserved.
//

#pragma once

#include <utility>
#include <vector>

namespace x {
  struct Event {
    Event() : x(0), y(0), ts(-1.0), polarity(false) {}
    Event(uint16_t _x, uint16_t _y, double _ts, bool _polarity)
    : x(_x), y(_y), ts(_ts), polarity(_polarity) {}
    uint16_t x;
    uint16_t y;
    // x library uses double for timestamps
    double ts;
    bool polarity;
  };

  struct EventArray {
    EventArray() : height(0), width(0), events() {};
    EventArray(uint32_t _height, uint32_t _width, std::vector<Event> _events)
    : height(_height), width(_width), events(std::move(_events)) {};
    uint32_t height;
    uint32_t width;
    std::vector<Event> events;

    typedef std::shared_ptr< ::x::EventArray> Ptr;
    typedef std::shared_ptr< ::x::EventArray const> ConstPtr;
  };
}