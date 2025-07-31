#define CATCH_CONFIG_MAIN
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
    constexpr double mu = 0.0;
    constexpr double kappa = 2.0;
    double sum = 0.0;
    for (int i = 0; i < N; ++i) {
      double x = -π + (2 * π) * i / N;
      sum += von_mises(mu, kappa, x);
    }
    double integral = sum * (2 * π / N);
    REQUIRE(integral == Approx(1.0).margin(1e-4));
  }

  SECTION("Symmetry: von_mises(mu, kappa, x) == von_mises(-mu, kappa, -x)") {
    constexpr double kappa = 1.5;
    constexpr double mu = 0.7;
    for (double x = -π; x <= π; x += 0.1) {
      double lhs = von_mises(mu, kappa, x);
      double rhs = von_mises(-mu, kappa, -x);
      REQUIRE(lhs == Approx(rhs).margin(1e-10));
    }
  }

  SECTION("Limiting case: kappa=0 gives uniform distribution") {
    constexpr double kappa = 0.0;
    constexpr double mu = 1.0;  // arbitrary
    for (double x = -π; x <= π; x += 0.1) {
      double pdf = von_mises(mu, kappa, x);
      REQUIRE(pdf == Approx(1.0 / (2 * π)).margin(1e-10));
    }
  }

  SECTION(
      "Maximum at mean: von_mises(mu, kappa, mu) >= von_mises(mu, kappa, x) for all x") {
    constexpr double kappa = 3.0;
    constexpr double mu = -0.5;
    double max_pdf = von_mises(mu, kappa, mu);
    for (double x = -π; x <= π; x += 0.1) {
      double pdf = von_mises(mu, kappa, x);
      REQUIRE(max_pdf >= pdf - 1e-12);
    }
  }
}
