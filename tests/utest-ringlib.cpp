#define CATCH_CONFIG_MAIN
#include <iostream>
#include <numbers>

#include "catch2/catch_all.hpp"
// #include "catch2/matchers/catch_matchers.hpp"
// #include "catch2/matchers/catch_matchers_range_equals.hpp"
//
#include "ringlib/RingAttractor.hpp"

using Catch::Approx;

constexpr double π = std::numbers::pi;

using Catch::Matchers::AllMatch;
using Catch::Matchers::RangeEquals;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

TEST_CASE("Function inter_neuron_distance sanity:") {
  SECTION("The distance between any neuron and itself is zero") {
    REQUIRE(ringlib::inter_neuron_distance<1>(0, 0) == 0);
    REQUIRE(ringlib::inter_neuron_distance<8>(5, 5) == 0);
    REQUIRE(ringlib::inter_neuron_distance<16>(8, 8) == 0);
  }
  SECTION("Should be invariant with swap of arguments") {
    REQUIRE(ringlib::inter_neuron_distance<16>(3, 12)
            == ringlib::inter_neuron_distance<16>(12, 3));
  }
  SECTION("Adjacent elements should be separated by 1, including on the ends") {
    REQUIRE(ringlib::inter_neuron_distance<3>(0, 1) == 1);
    REQUIRE(ringlib::inter_neuron_distance<3>(1, 2) == 1);
    REQUIRE(ringlib::inter_neuron_distance<3>(2, 0) == 1);
  }
  SECTION("Arbitrarily chosen cases done by hand") {
    REQUIRE(ringlib::inter_neuron_distance<16>(6, 12) == 6);
    REQUIRE(ringlib::inter_neuron_distance<32>(0, 31) == 1);
  }
}

TEST_CASE("Function angle_between basic sanity") {
  SECTION("Angle between any index and itself is 0°") {
    REQUIRE(ringlib::angle_between<11>(5, 5) == 0.);
    REQUIRE(ringlib::angle_between<1>(0, 0) == 0.);
    REQUIRE(ringlib::angle_between<8>(5, 5) == 0.);
    REQUIRE(ringlib::angle_between<16>(8, 8) == 0.);
  }
  SECTION("Angle from centre to edge is 180°") {
    REQUIRE(ringlib::angle_between<18>(0, 9) == π);
  }
  SECTION("Some obvious compass angles") {
    REQUIRE(ringlib::angle_between<4>(0, 1) == π / 2.);
    REQUIRE(ringlib::angle_between<4>(0, 3) == -π / 2.);
  }
  SECTION("Angles should change sign with argument flip") {
    REQUIRE(ringlib::angle_between<4>(1, 0) == -π / 2.);
  }
}

TEST_CASE("von_mises distribution properties") {
  using ringlib::von_mises;
  constexpr double tol = 1e-10;

  SECTION("Normalization: integral over [-pi, pi] is 1 (approximate by sum)") {
    constexpr int N = 10000;
    constexpr double μ = 0.0;
    constexpr double κ = 2.0;
    double sum = 0.0;
    for (int i = 0; i < N; ++i) {
      double x = -π + (2 * π) * i / N;
      sum += von_mises(μ, κ, x);
    }
    double integral = sum * (2 * π / N);
    REQUIRE(integral == Approx(1.0).margin(1e-4));
  }

  SECTION("Symmetry: von_mises(μ, κ, x) == von_mises(-μ, κ, -x)") {
    constexpr double κ = 1.5;
    constexpr double μ = 0.7;
    for (double x = -π; x <= π; x += 0.1) {
      double lhs = von_mises(μ, κ, x);
      double rhs = von_mises(-μ, κ, -x);
      REQUIRE(lhs == Approx(rhs).margin(1e-10));
    }
  }

  SECTION("Limiting case: κ=0 gives uniform distribution") {
    constexpr double κ = 0.0;
    constexpr double μ = 1.0;  // arbitrary
    for (double x = -π; x <= π; x += 0.1) {
      double pdf = von_mises(μ, κ, x);
      REQUIRE(pdf == Approx(1.0 / (2 * π)).margin(1e-10));
    }
  }

  SECTION("Maximum at mean: von_mises(μ, κ, μ) >= von_mises(μ, κ, x) for all x") {
    constexpr double κ = 3.0;
    constexpr double μ = -0.5;
    double max_pdf = von_mises(μ, κ, μ);
    for (double x = -π; x <= π; x += 0.1) {
      double pdf = von_mises(μ, κ, x);
      REQUIRE(max_pdf >= pdf - 1e-12);
    }
  }
}

TEST_CASE("Function `von_mises_input_single` basic sanity") {
  using ringlib::angle_of;
  using ringlib::von_mises_input_single;

  SECTION("Peak value is gamma for kappa > 0") {
    constexpr size_t N = 32;
    constexpr double κ = 2.0;
    constexpr double θ = π;
    constexpr double γ = 3.0;
    auto b = von_mises_input_single<N>(κ, θ, γ);
    REQUIRE(b.maxCoeff() == Approx(γ).margin(1e-10));
  }

  SECTION("Limiting case: k=0 gives flat input with height gamma") {
    constexpr size_t N = 8;
    constexpr double κ = 0.0;
    constexpr double θ = 0.0;
    constexpr double γ = 1.0;
    auto b = von_mises_input_single<N>(κ, θ, γ);
    for (size_t i = 0; i < N; ++i) {
      REQUIRE(b[i] == Approx(γ).margin(1e-10));
    }
  }

  SECTION("Maximum at mean: neuron closest to mu+theta has largest input") {
    constexpr size_t N = 32;
    constexpr double κ = 3.0;
    constexpr double θ = 0.5;
    constexpr double γ = 1.0;
    auto b = von_mises_input_single<N>(κ, θ, γ);
    size_t max_idx = 0;
    b.maxCoeff(&max_idx);
    double max_angle = angle_of<N>(max_idx);
    double expected_angle = θ;
    double diff = std::fmod(max_angle - expected_angle + π, 2 * π) - π;
    REQUIRE(std::abs(diff) < (2 * π / N));
  }

  SECTION("Single peak property: only one global maximum should exist") {
    constexpr size_t N = 18;  // Same as your ring size
    constexpr double κ = 7.0;  // Same as your κ
    constexpr double θ = π / 3;  // Arbitrary angle
    constexpr double γ = 8.0;  // Same as your γ
    auto b = von_mises_input_single<N>(κ, θ, γ);

    // Find global maximum
    size_t max_idx = 0;
    double max_val = b.maxCoeff(&max_idx);

    // Count how many neurons have values very close to the maximum
    int peak_count = 0;
    for (size_t i = 0; i < N; ++i) {
      if (b[i] > max_val * 0.99) {  // Within 1% of maximum
        peak_count++;
      }
    }

    // Should have only one peak (or at most two adjacent neurons due to discretization)
    REQUIRE(peak_count <= 2);

    // Find how many distinct local maxima exist
    std::vector<bool> is_local_max(N, false);
    for (size_t i = 0; i < N; ++i) {
      size_t prev = (i + N - 1) % N;
      size_t next = (i + 1) % N;
      if (b[i] > b[prev] && b[i] > b[next]) {
        is_local_max[i] = true;
      }
    }

    int local_max_count = std::count(is_local_max.begin(), is_local_max.end(), true);
    REQUIRE(local_max_count == 1);  // Should have exactly one local maximum
  }
}

TEST_CASE("von_mises_input_multi properties") {
  using ringlib::angle_of;
  using ringlib::von_mises_input_multi;
  constexpr double tol = 1e-10;

  SECTION("Single input matches von_mises_input_single") {
    constexpr size_t N = 16;
    constexpr double κ = 1.0;
    std::array<double, 1> θs = {0.5};
    std::array<double, 1> γs = {2.0};
    auto b_multi = von_mises_input_multi<N>(κ, std::span<const double>(θs),
                                            std::span<const double>(γs));
    auto b_single = ringlib::von_mises_input_single<N>(κ, θs[0], γs[0]);
    for (size_t i = 0; i < N; ++i) {
      REQUIRE(b_multi[i] == Approx(b_single[i]).margin(1e-10));
    }
  }

  SECTION("Sum of two inputs equals sum of singles") {
    constexpr size_t N = 16;
    constexpr double κ = 1.0;
    std::array<double, 2> θs = {π / 2, -π / 2};
    std::array<double, 2> γs = {1.0, 2.0};
    auto b_multi = von_mises_input_multi<N>(κ, std::span<const double>(θs),
                                            std::span<const double>(γs));
    auto b0 = ringlib::von_mises_input_single<N>(κ, θs[0], γs[0]);
    auto b1 = ringlib::von_mises_input_single<N>(κ, θs[1], γs[1]);
    for (size_t i = 0; i < N; ++i) {
      REQUIRE(b_multi[i] == Approx(b0[i] + b1[i]).margin(1e-10));
    }
  }

  SECTION("Limiting case: k=0 gives flat input with height sum(gammas)") {
    constexpr size_t N = 8;
    constexpr double κ = 0.0;
    std::array<double, 3> θs = {0.0, 1.0, 2.0};
    std::array<double, 3> γs = {1.0, 2.0, 3.0};
    auto b = von_mises_input_multi<N>(κ, std::span<const double>(θs),
                                      std::span<const double>(γs));
    double expected = 0.0;
    for (double g : γs)
      expected += g;
    for (size_t i = 0; i < N; ++i) {
      REQUIRE(b[i] == Approx(expected).margin(1e-10));
    }
  }
}
