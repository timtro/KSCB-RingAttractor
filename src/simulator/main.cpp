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
constexpr double γ = 2.0;
constexpr double κ = 20.0;
constexpr double ν = 0.5;
constexpr double network_coupling_constant = 1.5;

auto to_json(const RobotModel &robot) -> json {
  return json{{"x", robot.x()},
              {"y", robot.y()},
              {"θ", robot.θ()},
              {"v", robot.v()},
              {"ω", robot.ω()}};
}

template <size_t N>
auto to_json(const ringlib::FeleRingAttractor<N> &attractor) -> json {
  return json{{"neurons", attractor.neurons.transpose().eval()}};
}

ringlib::FeleRingAttractor<RING_SIZE> ring_attractor(ν, network_coupling_constant);
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
