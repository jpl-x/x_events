//
// Created by Florian Mahlknecht on 2021-03-25.
// Copyright (c) 2021 NASA / JPL. All rights reserved.


#pragma once

#include <string>
#include <fstream>
#include <limits>
#include <cassert>
#include <queue>
#include <tuple>
#include <easy/profiler.h>
#include <x/common/memory_monitor.h>

namespace x {

  /**
   * Recursive variadic template function which writes the values in CSV format to a file.
   * @tparam First type to be written on ostream
   * @tparam Types variadic templates waiting to be written
   * @param out ostream to work on (e.g. std::err or std::ofstream)
   * @param first value to be written on ostream
   * @param values values to be called recursively
   */
  template<class First, class... Types>
  void csv_write_row(std::ostream& out, First &&first, Types &&... values) {
    out << first;
    // evaluated at compile time --> C++17
    if constexpr (sizeof...(Types) > 0) {
      out << ";";
      x::csv_write_row(out, values...);
    } else {
      out << std::endl;
    }
  }

  template<class First, class... Types>
  size_t variadic_template_size(First &&first, Types &&... values) {
    auto size = sizeof(First);
    if constexpr (sizeof...(Types) > 0) {
      size += x::variadic_template_size(values...);
    }
    return size;
  }

  /**
   * High performance scientific CSV writer.
   * Provides a type-safe way to write CSV files through a templated addRow() method. These are written into a
   * queue to provide superior performance. Flush() writes them correctly formatted into a file.
   * All types are required to have a "<<" ostream operator.
   *
   * // Example usage:
   * x::CsvWriter<std::string, double, double> csv("test.csv", {"type", "x", "accuracy"});
   * csv.addRow("IMU", 1.6, 0.000004*(-1));
   *
   * @tparam Types types to be used in the
   */

  template<class ... Types>
  class CsvWriter : DebugMemory {
  public:
    /**
     * Opens the passed 'filename' as file in write mode and writes the provided column names in the constructor
     * directly to the file. They are required to have the same length as the variadic template types.
     * @param filename
     * @param column_names e.g. { "col1", "col2" }
     */
    CsvWriter(const std::string& filename, std::array<std::string, sizeof...(Types)> column_names);

    size_t memory_usage_in_bytes() const override;

    /**
     * Writes the buffer in CSV-format to the open file.
     */
    void flush() {
      while (!buffer_.empty()) {
        // expands variadic template types and forwards them to x::csv_write_row
        std::apply([=](auto &&... args) { x::csv_write_row(outfile_, args...); }, buffer_.front());
        buffer_.pop();
      }
      outfile_.flush();
    }

    void addRow(const Types &... values) {
      EASY_BLOCK("CSV addRow");
      buffer_.emplace(values...);
    }

    void addRow(Types &&... values) {
      EASY_BLOCK("CSV addRow");
      buffer_.emplace(values...);
    }

    ~CsvWriter() override {
      flush();
      outfile_.close();
    }

  private:
    std::queue<std::tuple<Types...>> buffer_;
    std::ofstream outfile_;
  };

  template<class... Types>
  CsvWriter<Types...>::CsvWriter(const std::string &filename, std::array<std::string, sizeof...(Types)> column_names)
    : DebugMemory() {
    outfile_.open(filename);

    // use double output precision
    outfile_.flags(std::ios::scientific);
    outfile_.precision(std::numeric_limits<double>::digits10);

    // place ; after each element except last
    auto it  = column_names.begin();
    for (;it < column_names.end()-1; ++it) {
      outfile_ << *it << ";";
    }
    outfile_ << *it << std::endl;
  }

  template<class... Types>
  size_t CsvWriter<Types...>::memory_usage_in_bytes() const {
    if (!buffer_.empty()) {
      auto lambda_expr = [=](auto &&... args) -> size_t { return x::variadic_template_size(args...); };

      return buffer_.size() * lambda_expr(buffer_.front());
    }
    return 0;
  }

}



