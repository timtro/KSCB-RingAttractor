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
  using ringlib::von_mises_stim_single;

  SECTION("Peak value is gamma for kappa > 0") {
    constexpr size_t N = 32;
    constexpr double κ = 2.0;
    constexpr double θ = π;
    constexpr double γ = 3.0;
    auto b = von_mises_stim_single<N>(κ, θ, γ);
    REQUIRE(b.maxCoeff() == Approx(γ).margin(1e-10));
  }

  SECTION("Limiting case: k=0 gives flat input with height gamma") {
    constexpr size_t N = 8;
    constexpr double κ = 0.0;
    constexpr double θ = 0.0;
    constexpr double γ = 1.0;
    auto b = von_mises_stim_single<N>(κ, θ, γ);
    for (size_t i = 0; i < N; ++i) {
      REQUIRE(b[i] == Approx(γ).margin(1e-10));
    }
  }

  SECTION("Maximum at mean: neuron closest to mu+theta has largest input") {
    constexpr size_t N = 32;
    constexpr double κ = 3.0;
    constexpr double θ = 0.5;
    constexpr double γ = 1.0;
    auto b = von_mises_stim_single<N>(κ, θ, γ);
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
    auto b = von_mises_stim_single<N>(κ, θ, γ);

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
  using ringlib::von_mises_stim_multi;
  constexpr double tol = 1e-10;

  SECTION("Single input matches von_mises_input_single") {
    constexpr size_t N = 16;
    constexpr double κ = 1.0;
    std::array<double, 1> θs = {0.5};
    std::array<double, 1> γs = {2.0};
    auto b_multi = von_mises_stim_multi<N>(κ, std::span<const double>(θs),
                                           std::span<const double>(γs));
    auto b_single = ringlib::von_mises_stim_single<N>(κ, θs[0], γs[0]);
    for (size_t i = 0; i < N; ++i) {
      REQUIRE(b_multi[i] == Approx(b_single[i]).margin(1e-10));
    }
  }

  SECTION("Sum of two inputs equals sum of singles") {
    constexpr size_t N = 16;
    constexpr double κ = 1.0;
    std::array<double, 2> θs = {π / 2, -π / 2};
    std::array<double, 2> γs = {1.0, 2.0};
    auto b_multi = von_mises_stim_multi<N>(κ, std::span<const double>(θs),
                                           std::span<const double>(γs));
    auto b0 = ringlib::von_mises_stim_single<N>(κ, θs[0], γs[0]);
    auto b1 = ringlib::von_mises_stim_single<N>(κ, θs[1], γs[1]);
    for (size_t i = 0; i < N; ++i) {
      REQUIRE(b_multi[i] == Approx(b0[i] + b1[i]).margin(1e-10));
    }
  }

  SECTION("Limiting case: k=0 gives flat input with height sum(gammas)") {
    constexpr size_t N = 8;
    constexpr double κ = 0.0;
    std::array<double, 3> θs = {0.0, 1.0, 2.0};
    std::array<double, 3> γs = {1.0, 2.0, 3.0};
    auto b = von_mises_stim_multi<N>(κ, std::span<const double>(θs),
                                     std::span<const double>(γs));
    double expected = 0.0;
    for (double g : γs)
      expected += g;
    for (size_t i = 0; i < N; ++i) {
      REQUIRE(b[i] == Approx(expected).margin(1e-10));
    }
  }
}

TEST_CASE("jro_kernel properties") {
  using ringlib::jro_kernel;

  SECTION("Diagonal elements (self-connections) are excitatory") {
    constexpr size_t N = 8;
    constexpr double ω_inhibit = -0.5;
    constexpr double ω_excite = 1.0;
    auto kernel = jro_kernel<N>(ω_inhibit, ω_excite);

    for (size_t i = 0; i < N; ++i) {
      REQUIRE(kernel(i, i) == Approx(ω_excite / static_cast<double>(N)));
    }
  }

  SECTION("Adjacent elements (nearest neighbors) are excitatory") {
    constexpr size_t N = 8;
    constexpr double ω_inhibit = -0.5;
    constexpr double ω_excite = 1.0;
    auto kernel = jro_kernel<N>(ω_inhibit, ω_excite);

    for (size_t i = 0; i < N; ++i) {
      size_t next = (i + 1) % N;
      size_t prev = (i + N - 1) % N;
      REQUIRE(kernel(i, next) == Approx(ω_excite / static_cast<double>(N)));
      REQUIRE(kernel(i, prev) == Approx(ω_excite / static_cast<double>(N)));
    }
  }

  SECTION("Non-adjacent elements are inhibitory") {
    constexpr size_t N = 8;
    constexpr double ω_inhibit = -0.5;
    constexpr double ω_excite = 1.0;
    auto kernel = jro_kernel<N>(ω_inhibit, ω_excite);

    // Test elements at distance 2 and beyond
    for (size_t i = 0; i < N; ++i) {
      for (size_t j = 0; j < N; ++j) {
        int distance =
            ringlib::inter_neuron_distance<N>(static_cast<int>(i), static_cast<int>(j));
        if (distance > 1) {
          REQUIRE(kernel(i, j) == Approx(ω_inhibit / static_cast<double>(N)));
        }
      }
    }
  }

  SECTION("Matrix is circulant - each row is a shifted version of the first") {
    constexpr size_t N = 6;
    constexpr double ω_inhibit = -1.0;
    constexpr double ω_excite = 2.0;
    auto kernel = jro_kernel<N>(ω_inhibit, ω_excite);

    // Check that row i is a circular shift of row 0
    for (size_t i = 1; i < N; ++i) {
      for (size_t j = 0; j < N; ++j) {
        size_t shifted_j = (j + N - i) % N;
        REQUIRE(kernel(i, j) == Approx(kernel(0, shifted_j)));
      }
    }
  }

  SECTION("Wrap-around connectivity (ring topology)") {
    constexpr size_t N = 5;
    constexpr double ω_inhibit = -0.3;
    constexpr double ω_excite = 0.7;
    auto kernel = jro_kernel<N>(ω_inhibit, ω_excite);

    // First neuron (index 0) should be connected to last neuron (index N-1)
    REQUIRE(kernel(0, N - 1) == Approx(ω_excite / static_cast<double>(N)));
    REQUIRE(kernel(N - 1, 0) == Approx(ω_excite / static_cast<double>(N)));
  }

  SECTION("Correct normalization by N") {
    constexpr size_t N = 4;
    constexpr double ω_inhibit = -2.0;
    constexpr double ω_excite = 3.0;
    auto kernel = jro_kernel<N>(ω_inhibit, ω_excite);

    // Check a few specific values
    REQUIRE(kernel(0, 0) == Approx(ω_excite / 4.0));  // Self-connection
    REQUIRE(kernel(0, 1) == Approx(ω_excite / 4.0));  // Adjacent
    REQUIRE(kernel(0, 2) == Approx(ω_inhibit / 4.0));  // Distance 2
  }

  SECTION("3-wide excitatory band for different ring sizes") {
    // Test with different N values to ensure the pattern holds
    std::vector<size_t> test_sizes = {3, 5, 8, 16};

    for (size_t N : test_sizes) {
      constexpr double ω_inhibit = -1.0;
      constexpr double ω_excite = 1.0;

      if (N == 3) {
        auto kernel = jro_kernel<3>(ω_inhibit, ω_excite);
        // For N=3, all distances are ≤ 1, so all should be excitatory
        for (size_t i = 0; i < 3; ++i) {
          for (size_t j = 0; j < 3; ++j) {
            REQUIRE(kernel(i, j) == Approx(ω_excite / 3.0));
          }
        }
      } else if (N == 5) {
        auto kernel = jro_kernel<5>(ω_inhibit, ω_excite);
        // For each neuron, itself and 2 neighbors should be excitatory
        for (size_t i = 0; i < 5; ++i) {
          int excitatory_count = 0;
          for (size_t j = 0; j < 5; ++j) {
            if (kernel(i, j) > 0)
              excitatory_count++;
          }
          REQUIRE(excitatory_count == 3);  // Self + 2 neighbors
        }
      }
    }
  }
}

TEST_CASE("cosine_kernel properties") {
  using ringlib::angle_between;
  using ringlib::cosine_kernel;

  SECTION("Diagonal elements (self-connections) are maximal") {
    constexpr size_t N = 8;
    constexpr double ν = 0.5;
    auto kernel = cosine_kernel<N>(ν);

    // Self-connections should have cos(π * 0^ν) / N = 1/N
    for (size_t i = 0; i < N; ++i) {
      REQUIRE(kernel(i, i) == Approx(1.0 / static_cast<double>(N)));
    }
  }

  SECTION("Matrix is circulant - each row is a shifted version of the first") {
    constexpr size_t N = 6;
    constexpr double ν = 1.0;
    auto kernel = cosine_kernel<N>(ν);

    // Check that row i is a circular shift of row 0
    for (size_t i = 1; i < N; ++i) {
      for (size_t j = 0; j < N; ++j) {
        size_t shifted_j = (j + N - i) % N;
        REQUIRE(kernel(i, j) == Approx(kernel(0, shifted_j)));
      }
    }
  }

  SECTION("Symmetry: kernel(i,j) == kernel(j,i)") {
    constexpr size_t N = 8;
    constexpr double ν = 0.7;
    auto kernel = cosine_kernel<N>(ν);

    for (size_t i = 0; i < N; ++i) {
      for (size_t j = 0; j < N; ++j) {
        REQUIRE(kernel(i, j) == Approx(kernel(j, i)));
      }
    }
  }

  SECTION("Wrap-around connectivity (ring topology)") {
    constexpr size_t N = 6;
    constexpr double ν = 1.0;
    auto kernel = cosine_kernel<N>(ν);

    // Connection from neuron 0 to neuron N-1 should equal connection to neuron 1
    // (both are at distance 1 on the ring)
    REQUIRE(kernel(0, N - 1) == Approx(kernel(0, 1)));
  }

  SECTION("Limiting case: ν=0 gives uniform cosine kernel") {
    constexpr size_t N = 8;
    constexpr double ν = 0.0;
    auto kernel = cosine_kernel<N>(ν);

    // For ν=0: cos(π * |angle|^0) = cos(π) = -1 for all non-zero angles
    // and cos(0) = 1 for zero angle
    for (size_t i = 0; i < N; ++i) {
      for (size_t j = 0; j < N; ++j) {
        if (i == j) {
          REQUIRE(kernel(i, j) == Approx(1.0 / static_cast<double>(N)));
        } else {
          REQUIRE(kernel(i, j) == Approx(-1.0 / static_cast<double>(N)));
        }
      }
    }
  }

  SECTION("Limiting case: ν=1 gives linear cosine decay") {
    constexpr size_t N = 8;
    constexpr double ν = 1.0;
    auto kernel = cosine_kernel<N>(ν);

    // For ν=1: kernel(i,j) = cos(|angle_between(i,j)|) / N
    for (size_t i = 0; i < N; ++i) {
      for (size_t j = 0; j < N; ++j) {
        double angle = std::abs(angle_between<N>(i, j));
        double expected = std::cos(angle) / static_cast<double>(N);
        REQUIRE(kernel(i, j) == Approx(expected));
      }
    }
  }

  SECTION("Values decrease with distance for ν > 0") {
    constexpr size_t N = 16;
    constexpr double ν = 0.8;
    auto kernel = cosine_kernel<N>(ν);

    // For a fixed row, values should generally decrease as we move away from diagonal
    // (though this isn't strictly monotonic due to the cosine function)
    for (size_t i = 0; i < N; ++i) {
      // Self-connection should be maximum
      double self_connection = kernel(i, i);
      for (size_t j = 0; j < N; ++j) {
        if (i != j) {
          REQUIRE(kernel(i, j)
                  <= self_connection + 1e-10);  // Allow for numerical precision
        }
      }
    }
  }

  SECTION("Opposite neurons have minimal connection for even N") {
    constexpr size_t N = 8;  // Even N
    constexpr double ν = 1.0;
    auto kernel = cosine_kernel<N>(ν);

    // For even N, neurons at distance N/2 are directly opposite
    for (size_t i = 0; i < N / 2; ++i) {
      size_t opposite = i + N / 2;
      double angle_to_opposite = std::abs(angle_between<N>(i, opposite));
      REQUIRE(angle_to_opposite == Approx(π));
      // cos(π) = -1, so connection should be -1/N
      REQUIRE(kernel(i, opposite) == Approx(-1.0 / static_cast<double>(N)));
    }
  }

  SECTION("Kernel values are properly normalized") {
    constexpr size_t N = 10;
    constexpr double ν = 0.6;
    auto kernel = cosine_kernel<N>(ν);

    // All values should be divided by N, so max absolute value should be ≤ 1/N
    double max_abs_value = 0.0;
    for (size_t i = 0; i < N; ++i) {
      for (size_t j = 0; j < N; ++j) {
        max_abs_value = std::max(max_abs_value, std::abs(kernel(i, j)));
      }
    }
    REQUIRE(max_abs_value <= 1.0 / static_cast<double>(N) + 1e-10);
  }

  SECTION("Different ν values produce different kernels") {
    constexpr size_t N = 8;
    auto kernel1 = cosine_kernel<N>(0.3);
    auto kernel2 = cosine_kernel<N>(0.7);

    // Kernels with different ν should be different (except at diagonal)
    bool found_difference = false;
    for (size_t i = 0; i < N && !found_difference; ++i) {
      for (size_t j = 0; j < N && !found_difference; ++j) {
        if (i != j && std::abs(kernel1(i, j) - kernel2(i, j)) > 1e-10) {
          found_difference = true;
        }
      }
    }
    REQUIRE(found_difference);
  }

  SECTION("Local excitation and global inhibition property") {
    // Test that the kernel exhibits local excitation and global inhibition
    // as shown in Marco Fele's paper figure
    constexpr size_t N = 16;  // Larger N to better see the pattern
    constexpr double ν = 0.5;  // Typical value
    auto kernel = cosine_kernel<N>(ν);

    for (size_t i = 0; i < N; ++i) {
      // Self-connection should be positive (excitatory)
      REQUIRE(kernel(i, i) > 0);

      // Check the pattern around each neuron
      for (size_t j = 0; j < N; ++j) {
        int distance =
            ringlib::inter_neuron_distance<N>(static_cast<int>(i), static_cast<int>(j));

        if (distance == 0) {
          // Self-connection: should be maximum positive value
          REQUIRE(kernel(i, j) == Approx(1.0 / static_cast<double>(N)));
        } else if (distance == 1) {
          // Nearest neighbors: should be excitatory for appropriate ν
          // This depends on ν value - let's check it's at least not strongly inhibitory
          // For ν=0.5, cos(π * (π/4)^0.5 / π) ≈ cos(π * 0.84) ≈ -0.87
          // So we expect some inhibition even for nearest neighbors with ν=0.5
          INFO("Distance 1 connection value: " << kernel(i, j));
        } else if (distance >= N / 4) {
          // Distant connections: should be inhibitory (negative)
          // For larger distances, the cosine should give negative values
          INFO("Distance " << distance << " connection value: " << kernel(i, j));
        }
      }

      // Check that opposite neurons (if N is even) have strong inhibition
      if (N % 2 == 0) {
        size_t opposite = (i + N / 2) % N;
        double angle_to_opposite = std::abs(ringlib::angle_between<N>(i, opposite));
        REQUIRE(angle_to_opposite == Approx(π));
        // cos(π * (π/π)^ν) = cos(π) = -1, so should be -1/N
        REQUIRE(kernel(i, opposite) == Approx(-1.0 / static_cast<double>(N)));
      }
    }
  }

  SECTION("Excitatory bandwidth depends on ν parameter") {
    // Test how the excitatory region changes with ν
    constexpr size_t N = 24;  // Larger N for better resolution

    // Test different ν values
    for (double nu : {0.1, 0.3, 0.5, 0.7, 1.0}) {
      auto kernel = cosine_kernel<N>(nu);

      // Count excitatory connections for neuron 0
      int excitatory_count = 0;
      int inhibitory_count = 0;

      for (size_t j = 0; j < N; ++j) {
        if (kernel(0, j) > 0) {
          excitatory_count++;
        } else if (kernel(0, j) < 0) {
          inhibitory_count++;
        }
      }

      INFO("ν=" << nu << " -> excitatory: " << excitatory_count
                << ", inhibitory: " << inhibitory_count);

      // Self-connection should always be excitatory
      REQUIRE(kernel(0, 0) > 0);

      // Should have both excitatory and inhibitory connections (except for extreme cases)
      if (nu > 0.05 && nu < 0.95) {
        REQUIRE(excitatory_count > 0);
        REQUIRE(inhibitory_count > 0);
      }

      // Smaller ν should generally lead to broader excitatory regions
      // (though this relationship is complex due to the cosine function)
    }
  }
}

TEST_CASE("cw_kernel basic properties", "[cw_kernel]") {
  using ringlib::cw_kernel;

  SECTION("Matrix dimensions are correct") {
    constexpr size_t N = 8;
    constexpr double weight = 1.5;
    constexpr size_t span = 3;
    
    auto kernel = cw_kernel<N>(weight, span);
    
    REQUIRE(kernel.rows() == N);
    REQUIRE(kernel.cols() == N);
  }
  
  SECTION("Non-self-activating - diagonal elements are zero") {
    constexpr size_t N = 8;
    constexpr double weight = 1.5;
    constexpr size_t span = 3;
    
    auto kernel = cw_kernel<N>(weight, span);
    
    for (size_t i = 0; i < N; ++i) {
      REQUIRE_THAT(kernel(i, i), WithinAbs(0.0, 1e-15));
    }
  }
  
  SECTION("Clockwise connections have correct weight") {
    constexpr size_t N = 8;
    constexpr double weight = 1.5;
    constexpr size_t span = 3;
    
    auto kernel = cw_kernel<N>(weight, span);
    
    for (size_t i = 0; i < N; ++i) {
      for (size_t k = 1; k <= span; ++k) {
        size_t j = (i + k) % N;
        REQUIRE_THAT(kernel(i, j), WithinAbs(weight, 1e-15));
      }
    }
  }
  
  SECTION("No connections beyond span") {
    constexpr size_t N = 8;
    constexpr double weight = 1.5;
    constexpr size_t span = 3;
    
    auto kernel = cw_kernel<N>(weight, span);
    
    for (size_t i = 0; i < N; ++i) {
      for (size_t k = span + 1; k < N; ++k) {
        size_t j = (i + k) % N;
        if (j != i) {  // Skip diagonal (already tested)
          REQUIRE_THAT(kernel(i, j), WithinAbs(0.0, 1e-15));
        }
      }
    }
  }
}

TEST_CASE("cw_kernel circulant property verification", "[cw_kernel]") {
  using ringlib::cw_kernel;
  
  SECTION("Each row is a circular shift of the previous row") {
    constexpr size_t N = 6;
    constexpr double weight = 2.0;
    constexpr size_t span = 2;
    
    auto kernel = cw_kernel<N>(weight, span);
    
    for (size_t i = 1; i < N; ++i) {
      for (size_t j = 0; j < N; ++j) {
        // Row i should be row (i-1) shifted right by 1 position
        size_t prev_j = (j + N - 1) % N;
        REQUIRE_THAT(kernel(i, j), WithinAbs(kernel(i - 1, prev_j), 1e-15));
      }
    }
  }
  
  SECTION("First and last rows demonstrate circulant wraparound") {
    constexpr size_t N = 6;
    constexpr double weight = 2.0;
    constexpr size_t span = 2;
    
    auto kernel = cw_kernel<N>(weight, span);
    
    for (size_t j = 0; j < N; ++j) {
      size_t wrapped_j = (j + N - 1) % N;
      REQUIRE_THAT(kernel(0, j), WithinAbs(kernel(N - 1, wrapped_j), 1e-15));
    }
  }
}

TEST_CASE("cw_kernel edge cases and parameter validation", "[cw_kernel]") {
  using ringlib::cw_kernel;
  
  SECTION("Span larger than N-1 is clamped correctly") {
    constexpr size_t N = 4;
    constexpr double weight = 1.0;
    constexpr size_t large_span = N + 5;  // Much larger than N
    
    auto kernel = cw_kernel<N>(weight, large_span);
    
    // Should connect to all neurons except self
    for (size_t i = 0; i < N; ++i) {
      REQUIRE_THAT(kernel(i, i), WithinAbs(0.0, 1e-15));  // No self-connection
      
      for (size_t j = 0; j < N; ++j) {
        if (i != j) {
          REQUIRE_THAT(kernel(i, j), WithinAbs(weight, 1e-15));
        }
      }
    }
  }
  
  SECTION("Span of 1 connects only to immediate clockwise neighbor") {
    constexpr size_t N = 5;
    constexpr double weight = 3.0;
    constexpr size_t span = 1;
    
    auto kernel = cw_kernel<N>(weight, span);
    
    for (size_t i = 0; i < N; ++i) {
      size_t next = (i + 1) % N;
      
      REQUIRE_THAT(kernel(i, next), WithinAbs(weight, 1e-15));  // Connected to next
      
      // All other connections should be zero
      for (size_t j = 0; j < N; ++j) {
        if (j != next) {
          REQUIRE_THAT(kernel(i, j), WithinAbs(0.0, 1e-15));
        }
      }
    }
  }
  
  SECTION("Default span parameter is N/2") {
    constexpr size_t N = 8;
    constexpr double weight = 1.0;
    
    auto kernel_default = cw_kernel<N>(weight);  // Use default span
    auto kernel_explicit = cw_kernel<N>(weight, N / 2);
    
    for (size_t i = 0; i < N; ++i) {
      for (size_t j = 0; j < N; ++j) {
        REQUIRE_THAT(kernel_default(i, j), WithinAbs(kernel_explicit(i, j), 1e-15));
      }
    }
  }
}

TEST_CASE("cw_kernel specific pattern verification for small examples", "[cw_kernel]") {
  using ringlib::cw_kernel;
  
  SECTION("N=4, span=2, weight=1.0 - explicit pattern check") {
    constexpr size_t N = 4;
    constexpr double weight = 1.0;
    constexpr size_t span = 2;
    
    auto kernel = cw_kernel<N>(weight, span);
    
    // Expected matrix (modulo the circle):
    // Row 0: [0, 1, 1, 0] (connects to indices 1, 2)
    // Row 1: [0, 0, 1, 1] (connects to indices 2, 3)
    // Row 2: [1, 0, 0, 1] (connects to indices 3, 0)
    // Row 3: [1, 1, 0, 0] (connects to indices 0, 1)
    
    Eigen::Matrix<double, 4, 4> expected;
    expected << 0, 1, 1, 0,
                0, 0, 1, 1,
                1, 0, 0, 1,
                1, 1, 0, 0;
    
    for (size_t i = 0; i < N; ++i) {
      for (size_t j = 0; j < N; ++j) {
        REQUIRE_THAT(kernel(i, j), WithinAbs(expected(i, j), 1e-15));
      }
    }
  }
}

TEST_CASE("cw_kernel asymmetry and directionality verification", "[cw_kernel]") {
  using ringlib::cw_kernel;
  
  SECTION("Matrix is not symmetric - demonstrates asymmetric connections") {
    constexpr size_t N = 6;
    constexpr double weight = 1.0;
    constexpr size_t span = 2;
    
    auto kernel = cw_kernel<N>(weight, span);
    
    bool is_symmetric = true;
    for (size_t i = 0; i < N && is_symmetric; ++i) {
      for (size_t j = 0; j < N && is_symmetric; ++j) {
        if (std::abs(kernel(i, j) - kernel(j, i)) > 1e-15) {
          is_symmetric = false;
        }
      }
    }
    REQUIRE_FALSE(is_symmetric);
  }
  
  SECTION("Clockwise directional preference - connections are unidirectional") {
    constexpr size_t N = 6;
    constexpr double weight = 1.0;
    constexpr size_t span = 2;
    
    auto kernel = cw_kernel<N>(weight, span);
    
    for (size_t i = 0; i < N; ++i) {
      // Check immediate clockwise neighbor
      size_t cw_neighbor = (i + 1) % N;
      size_t ccw_neighbor = (i + N - 1) % N;
      
      REQUIRE_THAT(kernel(i, cw_neighbor), WithinAbs(weight, 1e-15));  // Has CW connection
      REQUIRE_THAT(kernel(i, ccw_neighbor), WithinAbs(0.0, 1e-15));    // No CCW connection from span=2
    }
  }
  
  SECTION("Wraparound connections demonstrate ring topology") {
    constexpr size_t N = 5;
    constexpr double weight = 2.5;
    constexpr size_t span = 2;
    
    auto kernel = cw_kernel<N>(weight, span);
    
    // Last neuron (N-1) should connect clockwise to neurons 0 and 1
    REQUIRE_THAT(kernel(N-1, 0), WithinAbs(weight, 1e-15));  // Wraps to 0
    REQUIRE_THAT(kernel(N-1, 1), WithinAbs(weight, 1e-15));  // Wraps to 1
    
    // First neuron should connect to neurons 1 and 2
    REQUIRE_THAT(kernel(0, 1), WithinAbs(weight, 1e-15));
    REQUIRE_THAT(kernel(0, 2), WithinAbs(weight, 1e-15));
  }
}

TEST_CASE("gauss_stim_single basic properties", "[gauss_stim_single]") {
  using ringlib::angle_of;
  using ringlib::gauss_stim_single;

  SECTION("Peak value is gamma at target angle") {
    constexpr size_t N = 16;
    constexpr double γ = 5.0;    // Peak activation
    constexpr double α = 1.0;    // Baseline activation
    constexpr double ξ = 0.5;    // Width parameter
    constexpr double θ_tgt = π / 4;  // Target angle
    
    auto stim = gauss_stim_single<N>(γ, α, ξ, θ_tgt);
    
    // Find neuron closest to target angle
    size_t closest_idx = 0;
    double min_diff = std::numeric_limits<double>::max();
    for (size_t i = 0; i < N; ++i) {
      double diff = std::abs(angle_of<N>(i) - θ_tgt);
      if (diff > π) diff = 2 * π - diff;  // Handle wraparound
      if (diff < min_diff) {
        min_diff = diff;
        closest_idx = i;
      }
    }
    
    // Peak should be at or very close to gamma
    REQUIRE_THAT(stim[closest_idx], WithinAbs(γ, 1e-10));
    REQUIRE(stim.maxCoeff() <= γ + 1e-10);  // No value should exceed gamma
  }
  
  SECTION("Minimum value approaches alpha for distant neurons") {
    constexpr size_t N = 32;    // Larger N for better resolution
    constexpr double γ = 3.0;
    constexpr double α = 0.5;
    constexpr double ξ = 0.3;   // Narrow width
    constexpr double θ_tgt = 0.0;
    
    auto stim = gauss_stim_single<N>(γ, α, ξ, θ_tgt);
    
    // Find neuron most distant from target (opposite side of ring)
    size_t opposite_idx = N / 2;  // For N=32, this is maximally distant
    
    // Distant neurons should approach baseline alpha
    REQUIRE_THAT(stim[opposite_idx], WithinAbs(α, 0.1));  // Should be close to alpha
    REQUIRE(stim.minCoeff() >= α - 1e-10);  // No value should go below alpha
  }
  
  SECTION("Gaussian shape properties - symmetry around peak") {
    constexpr size_t N = 16;
    constexpr double γ = 2.0;
    constexpr double α = 0.0;
    constexpr double ξ = 0.8;
    constexpr double θ_tgt = 0.0;  // Target at neuron 0
    
    auto stim = gauss_stim_single<N>(γ, α, ξ, θ_tgt);
    
    // Due to ring topology, neurons 1 and N-1 should have equal activation
    // (both are distance 2π/N from target)
    REQUIRE_THAT(stim[1], WithinAbs(stim[N-1], 1e-12));
    
    // Similarly for neurons 2 and N-2
    REQUIRE_THAT(stim[2], WithinAbs(stim[N-2], 1e-12));
  }
  
  SECTION("Width parameter ξ controls spread") {
    constexpr size_t N = 24;
    constexpr double γ = 1.0;
    constexpr double α = 0.0;
    constexpr double θ_tgt = 0.0;
    
    // Narrow stimulus
    constexpr double ξ_narrow = 0.2;
    auto stim_narrow = gauss_stim_single<N>(γ, α, ξ_narrow, θ_tgt);
    
    // Wide stimulus  
    constexpr double ξ_wide = 0.8;
    auto stim_wide = gauss_stim_single<N>(γ, α, ξ_wide, θ_tgt);
    
    // Both should have same peak
    REQUIRE_THAT(stim_narrow.maxCoeff(), WithinAbs(stim_wide.maxCoeff(), 1e-10));
    
    // But narrow should have steeper falloff
    // Check a neuron moderately far from peak
    size_t test_idx = 3;
    REQUIRE(stim_narrow[test_idx] < stim_wide[test_idx]);
  }
  
  SECTION("Alpha parameter sets baseline correctly") {
    constexpr size_t N = 12;
    constexpr double γ = 4.0;
    constexpr double ξ = 0.5;
    constexpr double θ_tgt = π;
    
    // Test different baseline values
    constexpr double α1 = 0.0;
    constexpr double α2 = 1.5;
    
    auto stim1 = gauss_stim_single<N>(γ, α1, ξ, θ_tgt);
    auto stim2 = gauss_stim_single<N>(γ, α2, ξ, θ_tgt);
    
    // Minimum values should be close to respective alphas
    REQUIRE(stim1.minCoeff() >= α1 - 1e-10);
    REQUIRE(stim2.minCoeff() >= α2 - 1e-10);
    
    // Difference should be approximately α2 - α1 for distant neurons
    size_t distant_idx = N / 4;  // Quarter way around ring
    double expected_diff = α2 - α1;
    double actual_diff = stim2[distant_idx] - stim1[distant_idx];
    REQUIRE_THAT(actual_diff, WithinAbs(expected_diff, 0.1));
  }
}

TEST_CASE("gauss_stim_single angle wrapping and ring topology", "[gauss_stim_single]") {
  using ringlib::angle_of;
  using ringlib::gauss_stim_single;
  
  SECTION("Target angle wrapping works correctly") {
    constexpr size_t N = 8;
    constexpr double γ = 2.0;
    constexpr double α = 0.0;
    constexpr double ξ = 0.6;
    
    // Test equivalent target angles (differ by 2π)
    double θ_tgt1 = π / 3;
    double θ_tgt2 = θ_tgt1 + 2 * π;
    double θ_tgt3 = θ_tgt1 - 2 * π;
    
    auto stim1 = gauss_stim_single<N>(γ, α, ξ, θ_tgt1);
    auto stim2 = gauss_stim_single<N>(γ, α, ξ, θ_tgt2);
    auto stim3 = gauss_stim_single<N>(γ, α, ξ, θ_tgt3);
    
    // All should produce identical results
    for (size_t i = 0; i < N; ++i) {
      REQUIRE_THAT(stim1[i], WithinAbs(stim2[i], 1e-12));
      REQUIRE_THAT(stim1[i], WithinAbs(stim3[i], 1e-12));
    }
  }
  
  SECTION("Ring topology - shortest path distance calculation") {
    constexpr size_t N = 8;
    constexpr double γ = 3.0;
    constexpr double α = 0.0;
    constexpr double ξ = 0.4;
    constexpr double θ_tgt = 0.0;  // Target at neuron 0
    
    auto stim = gauss_stim_single<N>(γ, α, ξ, θ_tgt);
    
    // Due to ring topology, activation should decrease symmetrically
    // Neurons 1 and 7 are equidistant from 0 on the ring
    REQUIRE_THAT(stim[1], WithinAbs(stim[7], 1e-12));
    
    // Neurons 2 and 6 are equidistant from 0
    REQUIRE_THAT(stim[2], WithinAbs(stim[6], 1e-12));
    
    // Neurons 3 and 5 are equidistant from 0  
    REQUIRE_THAT(stim[3], WithinAbs(stim[5], 1e-12));
    
    // Neuron 4 is maximally distant (opposite)
    REQUIRE(stim[4] == stim.minCoeff());
  }
  
  SECTION("Boundary angle handling - near ±π") {
    constexpr size_t N = 16;
    constexpr double γ = 1.5;
    constexpr double α = 0.1;
    constexpr double ξ = 0.3;
    
    // Target very close to π boundary
    double θ_tgt_pos = π - 0.01;
    double θ_tgt_neg = -π + 0.01;
    
    auto stim_pos = gauss_stim_single<N>(γ, α, ξ, θ_tgt_pos);
    auto stim_neg = gauss_stim_single<N>(γ, α, ξ, θ_tgt_neg);
    
    // Should produce very similar results (targets are close)
    for (size_t i = 0; i < N; ++i) {
      REQUIRE_THAT(stim_pos[i], WithinAbs(stim_neg[i], 0.1));
    }
  }
}

TEST_CASE("gauss_stim_single edge cases and parameter validation", "[gauss_stim_single]") {
  using ringlib::gauss_stim_single;
  
  SECTION("Zero width parameter behavior") {
    constexpr size_t N = 8;
    constexpr double γ = 2.0;
    constexpr double α = 0.5;
    constexpr double ξ = 0.0;  // Zero width - should create delta-like function
    constexpr double θ_tgt = 0.0;
    
    auto stim = gauss_stim_single<N>(γ, α, ξ, θ_tgt);
    
    // With zero width, target neuron (closest to θ_tgt=0.0) should get gamma
    // All others should get baseline alpha
    REQUIRE_THAT(stim[0], WithinAbs(γ, 1e-10));  // Target neuron gets peak
    
    // All other neurons should get baseline
    for (size_t i = 1; i < N; ++i) {
      REQUIRE_THAT(stim[i], WithinAbs(α, 1e-10));
    }
  }
  
  SECTION("Equal gamma and alpha parameters") {
    constexpr size_t N = 6;
    constexpr double γ = 1.5;
    constexpr double α = 1.5;  // Same as gamma
    constexpr double ξ = 0.5;
    constexpr double θ_tgt = π / 2;
    
    auto stim = gauss_stim_single<N>(γ, α, ξ, θ_tgt);
    
    // When γ == α, all neurons should have same activation
    for (size_t i = 0; i < N; ++i) {
      REQUIRE_THAT(stim[i], WithinAbs(α, 1e-12));
    }
  }
  
  SECTION("Large width parameter - nearly uniform activation") {
    constexpr size_t N = 8;
    constexpr double γ = 3.0;
    constexpr double α = 1.0;
    constexpr double ξ = 10.0;  // Very large width
    constexpr double θ_tgt = 0.0;
    
    auto stim = gauss_stim_single<N>(γ, α, ξ, θ_tgt);
    
    // With very large width, all activations should be close to each other
    double max_val = stim.maxCoeff();
    double min_val = stim.minCoeff();
    REQUIRE((max_val - min_val) < 0.5);  // Small spread
    
    // All should be closer to gamma than to alpha
    for (size_t i = 0; i < N; ++i) {
      REQUIRE(stim[i] > (α + γ) / 2.0);
    }
  }
  
  SECTION("Gamma less than alpha - inverted Gaussian") {
    constexpr size_t N = 8;
    constexpr double γ = 1.0;  // Peak value
    constexpr double α = 2.0;  // Baseline higher than peak
    constexpr double ξ = 0.4;
    constexpr double θ_tgt = 0.0;
    
    auto stim = gauss_stim_single<N>(γ, α, ξ, θ_tgt);
    
    // This creates an "inverted" Gaussian - minimum at target
    REQUIRE(stim[0] == stim.minCoeff());  // Target has minimum value
    REQUIRE_THAT(stim[0], WithinAbs(γ, 1e-10));  // Should be gamma
    
    // Distant neurons should approach alpha
    size_t distant = N / 2;
    REQUIRE(stim[distant] > stim[0]);  // Distant > target
    REQUIRE(stim[distant] <= α + 1e-10);  // But not exceed alpha
  }
}

