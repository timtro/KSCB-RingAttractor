#pragma once

#include <Eigen/Dense>
#include <numbers>
#include <random>

namespace ringlib {

template <size_t N>
constexpr auto inter_neuron_distance(int i, int j) -> int {
  auto diff = std::abs(i - j);
  return std::min(diff, static_cast<int>(N) - diff);
}

template <size_t N>
constexpr auto angle_of(size_t i) -> double {
  using std::numbers::pi;
  return 2. * pi * static_cast<double>(i) / static_cast<double>(N);
}

template <size_t N>
constexpr auto angle_between(size_t i, size_t j) -> double {
  using std::numbers::pi;
  int diff = static_cast<int>(j) - static_cast<int>(i);
  int halfN = static_cast<int>(N) / 2;
  if (diff > halfN)
    diff -= static_cast<int>(N);
  if (diff < -halfN)
    diff += static_cast<int>(N);
  return 2. * pi * static_cast<double>(diff) / static_cast<double>(N);
}

// Marco Fele's kernel
template <size_t N>
constexpr auto cosine_kernel(double ν) {
  using std::numbers::pi;
  Eigen::Matrix<double, N, N> kernel = Eigen::Matrix<double, N, N>::Zero();
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      kernel(i, j) = std::cos(pi * std::pow(std::abs(angle_between<N>(i, j)) / pi, ν))
          / static_cast<double>(N);
    }
  }

  return kernel;
}

template <size_t N>
struct FeleRingAttractor {
  using VectorType = Eigen::Vector<double, N>;
  using MatrixType = Eigen::Matrix<double, N, N>;

  double ν;
  double network_coupling_constant;
  VectorType neurons;
  MatrixType weights;

  struct RandomVectorGenerator {
    std::mt19937 gen;
    std::normal_distribution<> dist;

    RandomVectorGenerator() : gen(std::random_device{}()), dist(0., 1E-4) {}

    VectorType operator()() {
      VectorType noise;
      for (int i = 0; i < N; ++i) {
        noise(i) = dist(gen) / static_cast<double>(N);
      }
      return noise;
    }
  };

  RandomVectorGenerator random_vector;

  FeleRingAttractor(double ν = 0.5, double network_coupling_constant = 1.)
      : ν(ν),
        network_coupling_constant(network_coupling_constant),
        neurons(VectorType::Zero()),
        weights(cosine_kernel<N>(ν)) {}

  void update(const VectorType &input, double dt) {
    // Ring attractor dynamics: dz/dt = -z + tanh(μWz + b) updated using Euler
    // integration.
    VectorType activity = network_coupling_constant * weights * neurons + input;
    neurons += dt * (-neurons + activity.array().tanh().matrix()) + random_vector();
  }

  auto state() const -> VectorType { return neurons; }

  auto heading() const -> double {
    using std::numbers::pi;
    double x_sum = 0, y_sum = 0;
    for (size_t i = 0; i < N; ++i) {
      double angle = angle_of<N>(i);
      x_sum += neurons[i] * std::cos(angle);
      y_sum += neurons[i] * std::sin(angle);
    }
    return std::atan2(y_sum, x_sum);
  }
};

// As per Marco Fele, use Von-Mises distribution function for use as shape for input
// signal.
constexpr auto von_mises(double μ, double κ, double x) -> double {
  using std::numbers::pi;

  // Normalize the difference to [-π, π]
  double diff = x - μ;
  diff = std::fmod(diff + pi, 2.0 * pi) - pi;

  // PDF: f(x; μ, κ) = exp(κ * cos(x - μ)) / (2π * I₀(κ))
  double numerator = std::exp(κ * std::cos(diff));
  double denominator = 2.0 * pi * std::cyl_bessel_i(0, κ);

  return numerator / denominator;
}

// Von Mises distribution function normalised such that the peak has height 1.
constexpr auto von_mises_peak_normalised(double μ, double κ, double x) -> double {
  using std::numbers::pi;

  // Normalize the difference to [-π, π]
  double diff = x - μ;
  diff = std::fmod(diff + pi, 2.0 * pi) - pi;

  // No longer PDF: f(x; μ, κ) = exp(κ * (cos(x - μ) - 1))
  return std::exp(κ * (std::cos(diff) - 1.0));
}

template <size_t N>
constexpr auto von_mises_input_single(double κ, double θ, double γ)
    -> Eigen::Vector<double, N> {
  Eigen::Vector<double, N> b;
  for (size_t i = 0; i < N; ++i) {
    b[i] = von_mises_peak_normalised(0, κ, angle_of<N>(i) - θ) * γ;
  }

  return b;
}

template <size_t N>
constexpr auto von_mises_input_multi(double κ,
                                     std::span<const double> θs,
                                     std::span<const double> γs)
    -> Eigen::Vector<double, N> {
  Eigen::Vector<double, N> b = Eigen::Vector<double, N>::Zero();
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < θs.size(); ++j) {
      b[i] += von_mises_peak_normalised(0, κ, angle_of<N>(i) - θs[j]) * γs[j];
    }
  }

  return b;
}

}  // namespace ringlib
