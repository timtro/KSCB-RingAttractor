#pragma once

#include <chrono>
#include <numbers>
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

struct JRONavParams {
  // Defaults from Rivero-Ortega et al (2023) doi:10.3389/fnbot.2023.1211570
  double τ = 0.001;  // Time constant of neuron dynamics.
  double γ = 100.;  // Scaling factor for Naka-Rushton function.
  double μ = 2.;  // exponent for Naka-Rushton function.
  double α = 5.0;  // minimal target ring stimulus.
  double σ = 40.;  // denominator constant term for Naka-Rushton function.
  double ξ = 1.2;  // SD of target input map Gaussian.
  double c = 120.0;  // c/ρ - controls rate of distance falloff of obstacle avoidance
                     // pressure.
  double ω_CA = 2.0;
  double ω_CB = 2.0;
  double ω_ED = 1.0;
  double ω_FC = 1.0;
  double ω_GE = 1.0;
  double ω_HF = 1.0;
  double ω_IG = 1.0;
  double ω_IH = 1.0;
  double ω_JH = 1.0;
  double ω_JG = 1.0;
  double ω_KI = 0.75;
  double ω_KJ = 0.75;
  double ω_U = 0.5;
  double ω_V = 1.0;
  double ω_S = 0.0125;
  double ω_EC = 1.0;
  double ω_FD = 1.0;
  double ω_CC_excite = 1.9;
  double ω_CC_inhibit = -1.7;
  double ω_DD_excite = 0.8;
  double ω_DD_inhibit = -0.9;
};

template <size_t N>
struct JRONavStack {
  using VectorType = Eigen::Vector<double, N>;
  using MatrixType = Eigen::Matrix<double, N, N>;

  double dt = 0.01;
  MotionState motion_state;

  auto x() const -> double { return motion_state(0); };
  auto y() const -> double { return motion_state(1); };
  auto θ() const -> double { return motion_state(2); };
  auto v() const -> double { return motion_state(3); };
  auto ω() const -> double { return motion_state(4); };

  JRONavParams params;

  ringlib::JRORing<N> target_ring;  // A
  ringlib::JRORing<N> obstacle_ring;  // B
  ringlib::JRORing<N> setpoint_ring;  // C
  ringlib::JRORing<N> orientation_ring;  // D
  ringlib::JRORing<N> error_ring_cw;  // E
  ringlib::JRORing<N> error_ring_ccw;  // F
  //
  ringlib::Neuron G{.τ = params.τ, .γ = params.γ, .σ = params.σ};
  ringlib::Neuron H{.τ = params.τ, .γ = params.γ, .σ = params.σ};
  ringlib::Neuron I{.τ = params.τ, .γ = params.γ, .σ = params.σ};
  ringlib::Neuron J{.τ = params.τ, .γ = params.γ, .σ = params.σ};
  ringlib::Neuron K{.τ = params.τ, .γ = params.γ, .σ = params.σ};

  MatrixType kernel_c =
      ringlib::jro_kernel<N>(params.ω_CC_inhibit, params.ω_CC_excite);  // C
  MatrixType kernel_d =
      ringlib::jro_kernel<N>(params.ω_DD_inhibit, params.ω_DD_excite);  // D
  MatrixType kernel_e = ringlib::cw_kernel<N>(params.ω_EC, N / 2);
  MatrixType kernel_f = ringlib::cw_kernel<N>(params.ω_FD, N / 2);

  void update_rings(Obstacle obstacle, Target target) {
    auto relative_obstacle = to_robot_polar(motion_state, obstacle);
    auto relative_target = to_robot_polar(motion_state, target);

    // A --- target encoding layer
    target_ring.update(
        ringlib::gauss_stim_single<N>(params.γ, params.α, params.ξ, relative_target.θ),
        dt);
    // B --- obstacle encoding layer
    double c_ρ = params.c / relative_obstacle.r;
    auto g = [this, &relative_obstacle](double c_over_ρ) -> VectorType {
      VectorType obstacle_stim = VectorType::Zero();

      // Find the neuron index closest to the obstacle angle
      double obstacle_angle = relative_obstacle.θ;
      size_t closest_neuron = 0;
      double min_angle_diff = std::abs(ringlib::angle_of<N>(0) - obstacle_angle);

      for (size_t i = 1; i < N; ++i) {
        double angle_diff = std::abs(ringlib::angle_of<N>(i) - obstacle_angle);
        if (angle_diff > std::numbers::pi) {
          angle_diff = 2.0 * std::numbers::pi - angle_diff;
        }
        if (angle_diff < min_angle_diff) {
          min_angle_diff = angle_diff;
          closest_neuron = i;
        }
      }

      // Apply piecewise stimulation to the closest neuron
      obstacle_stim[closest_neuron] = c_over_ρ >= params.γ ? params.γ : c_over_ρ;

      return obstacle_stim;
    };

    obstacle_ring.update(g(c_ρ));

    // C --- setpoint encoding layer
    setpoint_ring.update(params.ω_CA * target_ring.neurons
                             - params.ω_CB * obstacle_ring.neurons
                             + kernel_c * setpoint_ring.neurons,
                         dt);

    // D --- orientation encoding layer
    VectorType cos_vec;
    for (size_t i = 0; i < N; ++i) {
      cos_vec[i] = std::cos(ringlib::angle_of<N>(i) - θ());
    }
    orientation_ring.update(params.γ * cos_vec + kernel_d * orientation_ring.neurons, dt);
    // E and F --- CW and CCW error encoding layers
    error_ring_cw.update(
        params.ω_ED * orientation_ring.neurons + kernel_e * setpoint_ring.neurons, dt);
    error_ring_cw.update(
        params.ω_FC * setpoint_ring.neurons + kernel_f * orientation_ring.neurons, dt);
    //
    // Motor command circuit
    //
    // G
    G.update(params.ω_GE * error_ring_cw.neurons.sum());
    H.update(params.ω_HF * error_ring_ccw.neurons.sum());
    I.update(params.ω_IG * G.value - params.ω_IH * H.value, dt);
    J.update(params.ω_JH * H.value - params.ω_JG * G.value, dt);
    auto hBD = (params.ω_U * obstacle_ring.neurons.sum() * params.ω_V
                * orientation_ring.neurons.sum())
        / params.ω_S;
    constexpr double ρ_to = 100.00;
    K.update(-hBD - params.ω_KI * I.value - params.ω_KJ * J.value
                 + params.γ * relative_target.r / ρ_to,
             dt);
  }
};

auto rk4_step(const MotionState &x₀, const ControlSpace &u, double dt) -> MotionState;

}  // namespace rob
