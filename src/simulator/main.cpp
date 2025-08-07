#include <spdlog/spdlog.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <mutex>
#include <nlohmann/json.hpp>
#include <numbers>
#include <print>
#include <thread>

#include "ringlib/RingAttractor.hpp"
#include "simulator/RobotModel.hpp"

constexpr double π = std::numbers::pi;

using json = nlohmann::json;

constexpr double STEP_SIZE = 0.05;
constexpr size_t RING_SIZE = 18;

constexpr ringlib::FeleParameters fele_params{
    .ν = 0.5,  // Kernel parameter
    .υ = 1.5,  // Network coupling strength
    .γ = 2.0,  // Input gain
    .κ = 20.0  // Von Mises concentration
};

constexpr ringlib::JROParameters jro_params{
    // Defaults from Rivero-Ortega et al (2023) doi:10.3389/fnbot.2023.1211570
    .τ = 0.001,  // Time constant of neuron dynamics.
    .ω_inhibit = 1.9,  // (+) weight for excitatory connections in ring. NB: ω^{CC}
    .ω_excite = -1.7,  // (-) weight for inhibitory connections in ring.
    .γ = 100.,  // Scaling factor for Naka-Rushton function.
    .μ = 2.,  // exponent for Naka-Rushton function.
    .σ = 40.  // denominator constant term for Naka-Rushton function.
};

constexpr double γ = 2.0;
constexpr double κ = 20.0;
constexpr double ν = 0.5;
constexpr double υ = 1.5;

// auto to_json(const rob::RobotModel &robot) -> json {
//   return json{{"x", robot.x()},
//               {"y", robot.y()},
//               {"θ", robot.θ()},
//               {"v", robot.v()},
//               {"ω", robot.ω()}};
// }

template <size_t N>
auto to_json(const ringlib::FeleRingAttractor<N> &attractor) -> json {
  return json{{"neurons", attractor.neurons.transpose().eval()}};
}

ringlib::FeleRingAttractor<RING_SIZE> ring_attractor();

std::mutex global_state_mutex;
std::atomic<bool> running{true};

void simulation_loop() {
  while (running) {
    {
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
  }
}

auto main() -> int {
  spdlog::set_level(spdlog::level::info);
  spdlog::info("Starting Ring-attractor robot simulation");

  std::filesystem::path cwd = std::filesystem::current_path();
  spdlog::info("Current working directory: {}", cwd.string());

  spdlog::info("Spawing simulation thread");
  std::thread sim_thread(simulation_loop);

  running = false;
  sim_thread.join();
}
