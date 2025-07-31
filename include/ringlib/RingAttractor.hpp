#pragma once

#include <Eigen/Dense>
#include <numbers>

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

// template <size_t N>
// constexpr auto angle_between(size_t i, size_t j) -> double {
//   using std::numbers::pi;
//   auto diff = inter_neuron_distance<N>(i, j);
//   return 2. * pi * static_cast<double>(diff) / static_cast<double>(N);
// }

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
struct RingAttractor {
  using VectorType = Eigen::Vector<double, N>;
  using MatrixType = Eigen::Matrix<double, N, N>;

  double ν;
  VectorType neurons;
  MatrixType weights;

  RingAttractor(double ν = 0.5)
      : ν(ν), neurons(VectorType::Zero()), weights(cosine_kernel<N>(ν)) {}

  void update(const VectorType &inputs, double dt) {
    // Ring attractor dynamics: dz/dt = -z + tanh(Wz + inputs)
    VectorType activity = weights * neurons + inputs;
    neurons += dt * (-neurons + activity.array().tanh().matrix());
  }

  auto state() const -> VectorType { return neurons; }

  auto heading() const -> double {
    using std::numbers::pi;
    // Decode heading using population vector method
    double x_sum = 0, y_sum = 0;
    for (size_t i = 0; i < N; ++i) {
      double angle = 2 * pi * i / N;
      x_sum += neurons[i] * std::cos(angle);
      y_sum += neurons[i] * std::sin(angle);
    }
    return std::atan2(y_sum, x_sum);
  }
};

constexpr auto von_mises(double μ, double κ, double x) -> double {
  using std::numbers::pi;

  // Normalize the difference to [-π, π]
  double diff = x - μ;
  diff = std::fmod(diff + pi, 2.0 * pi) - pi;

  // PDF: f(x|μ,κ) = exp(κ*cos(x-μ)) / (2π * I₀(κ))
  double numerator = std::exp(κ * std::cos(diff));
  double denominator = 2.0 * pi * std::cyl_bessel_i(0, κ);

  return numerator / denominator;
}

}  // namespace ringlib
