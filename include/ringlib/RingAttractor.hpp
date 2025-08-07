#pragma once

#include <emmintrin.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cstddef>
#include <numbers>
#include <random>
#include <stdexcept>

namespace ringlib {

template <size_t N>
struct RandomVectorGenerator {
  std::mt19937 gen;
  std::normal_distribution<> dist;

  RandomVectorGenerator(double μ = 0., double σ = 1E-4)
      : gen(std::random_device{}()), dist(μ, σ) {}

  auto operator()() -> Eigen::Vector<double, N> {
    Eigen::Vector<double, N> noise;
    for (int i = 0; i < N; ++i) {
      noise(i) = dist(gen) / static_cast<double>(N);
    }
    return noise;
  }
};

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
      double abs_angle = std::abs(angle_between<N>(i, j));
      if (abs_angle == 0.0) {
        // Same neuron: cos(0) = 1
        kernel(i, j) = 1.0 / static_cast<double>(N);
      } else {
        // Different neurons: use the cosine kernel formula
        kernel(i, j) =
            std::cos(pi * std::pow(abs_angle / pi, ν)) / static_cast<double>(N);
      }
    }
  }

  return kernel;
}

struct FeleParameters {
  double ν = 0.5;  // Kernel parameter
  double υ = 2.6;  // Network coupling strength
                   // Ranges: 2.4 - unstable, 2.5 - marginally stable, 2.6 - stable
  double γ = 2.0;  // Input gain
  double κ = 20.0;  // Von Mises concentration
};

template <size_t N>
struct FeleRingAttractor {
  using VectorType = Eigen::Vector<double, N>;
  using MatrixType = Eigen::Matrix<double, N, N>;

  double ν;
  double υ;
  FeleParameters params;
  VectorType neurons;
  MatrixType weights;

  RandomVectorGenerator<N> random_vector;

  FeleRingAttractor(FeleParameters params_)
      : params(params_),
        neurons(VectorType::Zero()),
        weights(cosine_kernel<N>(params_.ν)) {}

  void update(const VectorType &input, double dt) {
    // Ring attractor dynamics: dz/dt = -z + tanh(υWz + b) updated using Euler
    // integration.
    VectorType activity = params.υ * weights * neurons + input;
    neurons += dt * (-neurons + activity.array().tanh().matrix()) + random_vector();
  }

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

template <size_t N>
struct FeleRing {
  using VectorType = Eigen::Vector<double, N>;
  using MatrixType = Eigen::Matrix<double, N, N>;

  VectorType neurons;

  FeleRing() : neurons(VectorType::Zero()) {}

  void update(const VectorType &input, double dt) {
    neurons += dt * (-neurons + input.array().tanh().matrix());
  }

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

// Naka-Rushton function for neural activation
constexpr auto naka_rushton(double γ, double μ, double σ, double z) -> double {
  if (z <= 0.0)
    return 0.0;

  double z_to_μ = std::pow(z, μ);
  double σ_to_μ = std::pow(σ, μ);
  return γ * z_to_μ / (z_to_μ + σ_to_μ);
}

// Optimized Naka-Rushton function for μ=2 (The case in JRO paper)
constexpr auto naka_rushton_μ2(double γ, double σ, double z) -> double {
  if (z <= 0.0)
    return 0.0;

  double z_sq = z * z;
  double σ_sq = σ * σ;
  return γ * z_sq / (z_sq + σ_sq);
}

template <size_t N>
constexpr auto jro_kernel(double ω_inhibit, double ω_excite)
    -> Eigen::Matrix<double, N, N> {
  Eigen::Matrix<double, N, N> kernel = Eigen::Matrix<double, N, N>::Zero();
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      // 3-wide diagonal is ω_excite, all other ω_inhibit:
      int distance = inter_neuron_distance<N>(static_cast<int>(i), static_cast<int>(j));
      kernel(i, j) = (distance <= 1) ? ω_excite : ω_inhibit;
    }
  }
  return kernel / static_cast<double>(N);
}

struct JROParameters {
  // Defaults from Rivero-Ortega et al (2023) doi:10.3389/fnbot.2023.1211570
  double τ = 0.001;  // Time constant of neuron dynamics.
  double ω_inhibit = 1.9;  // (+) weight for excitatory connections in ring. NB: ω^{CC}
  double ω_excite = -1.7;  // (-) weight for inhibitory connections in ring.
  double γ = 100.;  // Scaling factor for Naka-Rushton function.
  double μ = 2.;  // exponent for Naka-Rushton function.
  double σ = 40.;  // denominator constant term for Naka-Rushton function.
};

template <size_t N>
struct JRORingAttractor {
  using VectorType = Eigen::Vector<double, N>;
  using MatrixType = Eigen::Matrix<double, N, N>;

  JROParameters params;
  VectorType neurons;
  MatrixType weights;

  RandomVectorGenerator<N> random_vector;

  JRORingAttractor(JROParameters params_)
      : params(params_),
        neurons(VectorType::Zero()),
        weights(jro_kernel<N>(params_.ω_inhibit, params_.ω_excite)) {
    if (params.τ <= 0.0)
      throw std::invalid_argument("Time constant τ must be positive");
    if (params.σ <= 0.0)
      throw std::invalid_argument("NR sigma σ must be positive");
    if (params.γ <= 0.0)
      throw std::invalid_argument("NR gamma γ must be positive");
    if (params.μ <= 0.0)
      throw std::invalid_argument("NR exponent μ must be positive");
  }

  void update(const VectorType &input, double dt) {
    // Ring attractor dynamics: τ dy/dt = -y + NR(Wz + b) updated using Euler
    // integration.
    VectorType nr_activity = (weights * neurons + input).unaryExpr([this](double x) {
      return naka_rushton_μ2(params.γ, params.σ, x);
    });

    neurons += (dt / params.τ) * (-neurons + nr_activity) + random_vector();
  }

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

template <size_t N>
struct JRORing {
  using VectorType = Eigen::Vector<double, N>;
  using MatrixType = Eigen::Matrix<double, N, N>;

  double τ = 0.001;
  double γ = 100.0;
  double μ = 2;
  double σ = 40.0;
  VectorType neurons;

  JRORing(double τ_, double γ_, double μ_, double σ_)
      : τ(τ_), γ(γ_), μ(μ_), σ(σ_), neurons(VectorType::Zero()) {
    if (τ_ <= 0.0)
      throw std::invalid_argument("Time constant τ must be positive");
    if (σ_ <= 0.0)
      throw std::invalid_argument("NR sigma σ must be positive");
    if (γ_ <= 0.0)
      throw std::invalid_argument("NR gamma γ must be positive");
    if (μ_ != 2.0)
      throw std::invalid_argument(
          "This structure is specifically optimised. NR exponent, μ, must be 2.");
    if (μ_ <= 0.0)
      throw std::invalid_argument("NR exponent μ must be positive");
  }

  JRORing(JROParameters params_)
      : τ(params_.τ),
        γ(params_.γ),
        μ(params_.μ),
        σ(params_.σ),
        neurons(VectorType::Zero()) {
    if (τ <= 0.0)
      throw std::invalid_argument("Time constant τ must be positive");
    if (σ <= 0.0)
      throw std::invalid_argument("NR sigma σ must be positive");
    if (γ <= 0.0)
      throw std::invalid_argument("NR gamma γ must be positive");
    if (μ != 2.0)
      throw std::invalid_argument(
          "This structure is specifically optimised. NR exponent, μ, must be 2.");
    if (μ <= 0.0)
      throw std::invalid_argument("NR exponent μ must be positive");
  }

  void update(const VectorType &input, double dt) {
    VectorType nr_input =
        (input).unaryExpr([this](double x) { return naka_rushton_μ2(γ, σ, x); });

    neurons += (dt / τ) * (-neurons + nr_input);
  }

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
  double angle_diff = x - μ;
  angle_diff = std::fmod(angle_diff + pi, 2.0 * pi) - pi;
  angle_diff = std::fmod(angle_diff + pi, 2.0 * pi) - pi;
  // Handle edge cases where fmod might not behave consistently
  while (angle_diff <= -pi)
    angle_diff += 2.0 * pi;
  while (angle_diff > pi)
    angle_diff -= 2.0 * pi;

  // PDF: f(x; μ, κ) = exp(κ * cos(x - μ)) / (2π * I₀(κ))
  double numerator = std::exp(κ * std::cos(angle_diff));
  double denominator = 2.0 * pi * std::cyl_bessel_i(0, κ);

  return numerator / denominator;
}

// Von Mises distribution function normalised such that the peak has height 1.
constexpr auto von_mises_peak_normalised(double μ, double κ, double x) -> double {
  using std::numbers::pi;

  // Normalize the difference to [-π, π]
  double angle_diff = x - μ;
  angle_diff = std::fmod(angle_diff + pi, 2.0 * pi) - pi;
  angle_diff = std::fmod(angle_diff + pi, 2.0 * pi) - pi;
  // Handle edge cases where fmod might not behave consistently
  while (angle_diff <= -pi)
    angle_diff += 2.0 * pi;
  while (angle_diff > pi)
    angle_diff -= 2.0 * pi;

  // No longer PDF: f(x; μ, κ) = exp(κ * (cos(x - μ) - 1))
  return std::exp(κ * (std::cos(angle_diff) - 1.0));
}

template <size_t N>
constexpr auto von_mises_stim_single(double κ, double θ, double γ)
    -> Eigen::Vector<double, N> {
  Eigen::Vector<double, N> b;
  for (size_t i = 0; i < N; ++i) {
    b[static_cast<int>(i)] = von_mises_peak_normalised(0, κ, angle_of<N>(i) - θ) * γ;
  }

  return b;
}

template <size_t N>
constexpr auto von_mises_stim_multi(double κ,
                                    std::span<const double> θs,
                                    std::span<const double> γs)
    -> Eigen::Vector<double, N> {
  Eigen::Vector<double, N> b = Eigen::Vector<double, N>::Zero();
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < θs.size(); ++j) {
      b[static_cast<int>(i)] +=
          von_mises_peak_normalised(0, κ, angle_of<N>(i) - θs[j]) * γs[j];
    }
  }

  return b;
}

template <size_t N>
auto cw_kernel(double weight, size_t span = N / 2) -> Eigen::Matrix<double, N, N> {
  Eigen::Matrix<double, N, N> mat = Eigen::Matrix<double, N, N>::Zero();

  // Ensure span is less than N
  span = std::min(span, N - 1);

  for (size_t i = 0; i < N; ++i) {
    for (size_t k = 1; k <= span; ++k) {  // Start at 1 to EXCLUDE self
      size_t j = (i + k) % N;
      mat(i, j) = weight;
    }
  }

  return mat;
}

// TODO: Destroy when confirmed unneeded.
// template <size_t N>
// auto ccw_kernel(double weight, size_t span = N / 2) -> Eigen::Matrix<double, N, N> {
//   Eigen::Matrix<double, N, N> mat = Eigen::Matrix<double, N, N>::Zero();
//
//   // Ensure span is less than N
//   span = std::min(span, N - 1);
//
//   for (size_t i = 0; i < N; ++i) {
//     for (size_t k = 1; k <= span; ++k) {  // Start at 1 to EXCLUDE self
//       size_t j = (i + N - k) % N;
//       mat(i, j) = weight;
//     }
//   }
//
//   return mat;
// }

template <size_t N>
constexpr auto gauss_stim_single(double γ, double α, double ξ, double θ_tgt)
    -> Eigen::Vector<double, N> {
  using std::numbers::pi;
  Eigen::Vector<double, N> b;
  for (size_t i = 0; i < N; ++i) {
    double angle_diff = angle_of<N>(i) - θ_tgt;

    angle_diff = std::fmod(angle_diff + pi, 2.0 * pi) - pi;
    // Handle edge cases where fmod might not behave consistently
    while (angle_diff <= -pi)
      angle_diff += 2.0 * pi;
    while (angle_diff > pi)
      angle_diff -= 2.0 * pi;

    // Handle zero width parameter to avoid division by zero
    if (ξ == 0.0) {
      // Delta-like function: full activation at target, baseline elsewhere
      b[static_cast<int>(i)] = (std::abs(angle_diff) < 1e-10) ? γ : α;
    } else {
      b[static_cast<int>(i)] =
          α + (γ - α) * std::exp(-angle_diff * angle_diff / (2. * ξ * ξ));
    }
  }

  return b;
}

struct Neuron {
  double value = 0.;
  double τ = 0.;
  double γ = 0.;
  double σ = 0.;

  // Neuron(JROParameters params) : τ(params.τ), γ(params.γ), σ(params.σ) {
  //   if (τ <= 0.0)
  //     throw std::invalid_argument("Time constant τ must be positive");
  //   if (σ <= 0.0)
  //     throw std::invalid_argument("NR sigma σ must be positive");
  //   if (γ <= 0.0)
  //     throw std::invalid_argument("NR gamma γ must be positive");
  // }

  void update(double input, double dt) { value += dt * naka_rushton_μ2(γ, σ, input); }
};

}  // namespace ringlib

constexpr auto wrap_angle(double angle) -> double {
  // Wrap angle to [-π, π)
  using std::numbers::pi;
  angle = std::fmod(angle + pi, 2.0 * pi);
  while (angle < 0)
    angle += 2.0 * pi;
  return angle - pi;
}
