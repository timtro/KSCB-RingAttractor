#define CATCH_CONFIG_MAIN
#include <numbers>

#include "catch2/catch_all.hpp"
#include "catch2/matchers/catch_matchers_range_equals.hpp"
//
#include "ringlib/RingAttractor.hpp"

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
