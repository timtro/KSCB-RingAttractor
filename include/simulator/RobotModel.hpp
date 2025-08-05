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

using MotionState = Eigen::Matrix<double, 5, 1>;
using ControlSpace = Eigen::Matrix<double, 2, 1>;

auto by_elements(MotionState &x)
    -> std::tuple<double &, double &, double &, double &, double &>;

auto by_elements(const MotionState &x) -> std::
    tuple<const double &, const double &, const double &, const double &, const double &>;

template <size_t N>
struct RingAttractorNavStack {
  using VectorType = Eigen::Vector<double, N>;
  using MatrixType = Eigen::Matrix<double, N, N>;

  double dt = 0.01;
  MotionState motion_state;

  auto x() const -> double { return motion_state(0); };
  auto y() const -> double { return motion_state(1); };
  auto θ() const -> double { return motion_state(2); };
  auto v() const -> double { return motion_state(3); };
  auto ω() const -> double { return motion_state(4); };

  struct RingAttractorParameters {
    float γ = 8.0f;  // Input gain
    float κ = 8.0f;  // Von Mises concentration
    float ν = 0.10f;  // Kernel parameter
    float network_coupling_constant = 6.0f;  // Network coupling strength
  };

  RingAttractorParameters params;

  ringlib::FeleRingAttractor<N> target_ring;  // A
  ringlib::FeleRingAttractor<N> obstacle_ring;  // B
  ringlib::FeleRingAttractor<N> setpoint_ring;  // C
  ringlib::FeleRingAttractor<N> orientation_ring;  // D
  ringlib::FeleRingAttractor<N> error_ring;  // E and F

  void update_rings(std::span<Obstacle> obstacles, Target target) {}
};

auto rk4_step(const MotionState &x₀, const ControlSpace &u, double dt) -> MotionState;

}  // namespace rob
