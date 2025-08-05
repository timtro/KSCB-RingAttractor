#pragma once

#include <Eigen/Dense>
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
        // For same neuron, always cos(0) = 1
        kernel(i, j) = 1.0 / static_cast<double>(N);
      } else {
        kernel(i, j) = std::cos(pi * std::pow(abs_angle / pi, ν)) / static_cast<double>(N);
      }
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

  RandomVectorGenerator<N> random_vector;

  FeleRingAttractor(double ν_ = 0.5, double network_coupling_constant_ = 1.)
      : ν(ν_),
        network_coupling_constant(network_coupling_constant_),
        neurons(VectorType::Zero()),
        weights(cosine_kernel<N>(ν)) {}

  void update(const VectorType &input, double dt) {
    // Ring attractor dynamics: dz/dt = -z + tanh(μWz + b) updated using Euler
    // integration.
    VectorType activity = network_coupling_constant * weights * neurons + input;
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
// Optimized Naka-Rushton function for μ=2 (common case in JRO paper)
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

template <size_t N>
struct JRORingAttractor {
  using VectorType = Eigen::Vector<double, N>;
  using MatrixType = Eigen::Matrix<double, N, N>;

  double ω_excite, ω_inhibit, τ, γ, μ, σ;
  VectorType neurons;
  MatrixType weights;

  RandomVectorGenerator<N> random_vector;

  JRORingAttractor(double τ_,
                   double ω_excite_,
                   double ω_inhibit_,
                   double γ_,
                   double μ_,
                   double σ_)
      : τ(τ_),
        ω_excite(ω_excite_),
        ω_inhibit(ω_inhibit_),
        γ(γ_),
        μ(μ_),
        σ(σ_),
        neurons(VectorType::Zero()),
        weights(jro_kernel<N>(ω_inhibit_, ω_excite_)) {

    // Parameter validation
    if (τ_ <= 0.0)
      throw std::invalid_argument("Time constant τ must be positive");
    if (σ_ <= 0.0)
      throw std::invalid_argument("NR sigma σ must be positive");
    if (γ_ <= 0.0)
      throw std::invalid_argument("NR gamma γ must be positive");
    if (μ_ <= 0.0)
      throw std::invalid_argument("NR exponent μ must be positive");
  }

  void update(const VectorType &input, double dt) {
    // Ring attractor dynamics: τ dy/dt = -y + NR(Wz + b) updated using Euler
    // integration.
    VectorType nr_activity = (weights * neurons + input).unaryExpr([this](double x) {
      return naka_rushton_μ2(γ, σ, x);
    });

    neurons += (dt / τ) * (-neurons + nr_activity) + random_vector();
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
struct NRORing {
  using VectorType = Eigen::Vector<double, N>;
  using MatrixType = Eigen::Matrix<double, N, N>;

  double τ, γ, μ, σ;
  VectorType neurons;

  NRORing(double τ_, double γ_, double μ_, double σ_)
      : τ(τ_), γ(γ_), μ(μ_), σ(σ_), neurons(VectorType::Zero()) {

    // Parameter validation
    if (τ_ <= 0.0)
      throw std::invalid_argument("Time constant τ must be positive");
    if (σ_ <= 0.0)
      throw std::invalid_argument("NR sigma σ must be positive");
    if (γ_ <= 0.0)
      throw std::invalid_argument("NR gamma γ must be positive");
    if (μ_ <= 0.0)
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
    b[static_cast<int>(i)] = von_mises_peak_normalised(0, κ, angle_of<N>(i) - θ) * γ;
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
      b[static_cast<int>(i)] +=
          von_mises_peak_normalised(0, κ, angle_of<N>(i) - θs[j]) * γs[j];
    }
  }

  return b;
}

}  // namespace ringlib

constexpr auto wrap_angle(double angle) -> double {
  // Wrap angle to [-π, π)
  using std::numbers::pi;
  angle = std::fmod(angle + pi, 2.0 * pi);
  while (angle < 0)
    angle += 2.0 * pi;
  return angle - pi;
}
