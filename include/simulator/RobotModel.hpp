#pragma once

#include <chrono>
#include <tuple>
//
#include "ringlib/RingAttractor.hpp"

namespace rob {

struct Target {
  double x, y;
};

struct Obstacle {
  double x, y;
};

struct PolarCoordinates {
  double r;  // radius in meters
  double θ;  // angle in radians
};

using MotionState = Eigen::Matrix<double, 5, 1>;
using ControlSpace = Eigen::Matrix<double, 2, 1>;

auto by_elements(MotionState &x)
    -> std::tuple<double &, double &, double &, double &, double &>;

auto by_elements(const MotionState &x) -> std::
    tuple<const double &, const double &, const double &, const double &, const double &>;

template <typename T>
auto to_robot_polar(MotionState &m, T &t) -> PolarCoordinates {
  auto [x, y, θ, v, ω] = by_elements(m);
  auto Δx = x - t.x;
  auto Δy = y - t.y;
  return {.r = std::hypot(Δx, Δy), .θ = std::atan2(Δy, Δx)};
}

template <size_t N>
struct FeleAttractorNavStack {
  using VectorType = Eigen::Vector<double, N>;
  using MatrixType = Eigen::Matrix<double, N, N>;

  double dt = 0.01;
  MotionState motion_state;

  auto x() const -> double { return motion_state(0); };
  auto y() const -> double { return motion_state(1); };
  auto θ() const -> double { return motion_state(2); };
  auto v() const -> double { return motion_state(3); };
  auto ω() const -> double { return motion_state(4); };

  ringlib::FeleParameters params;

  ringlib::FeleRing<N> target_ring;  // A
  ringlib::FeleRing<N> obstacle_ring;  // B
  ringlib::FeleRingAttractor<N> setpoint_ring;  // C
  ringlib::FeleRingAttractor<N> orientation_ring;  // D
  ringlib::FeleRingAttractor<N> error_ring;  // E and F

  void update_rings(std::span<Obstacle> obstacles, Target target) {}
};

auto rk4_step(const MotionState &x₀, const ControlSpace &u, double dt) -> MotionState;

}  // namespace rob
