#include "simulator/RobotModel.hpp"

#include <Eigen/Dense>
#include <numbers>
#include <random>

namespace rob {

auto by_elements(RobotState &x)
    -> std::tuple<double &, double &, double &, double &, double &> {
  return std::tie(x(0), x(1), x(2), x(3), x(4));
}

auto by_elements(const RobotState &x) -> std::tuple<const double &,
                                                    const double &,
                                                    const double &,
                                                    const double &,
                                                    const double &> {
  return std::tie(x(0), x(1), x(2), x(3), x(4));
}

auto rk4_step(const RobotState &x₀, const ControlSpace &u, double dt) -> RobotState {
  auto dxdt = [&u](const RobotState &state) -> RobotState {
    const auto &[x, y, θ, v, ω] = by_elements(state);
    return {v * std::cos(θ),  // dx/dt = v * cos(θ)
            v * std::sin(θ),  // dy/dt = v * sin(θ)
            ω,  // dθ/dt = ω
            u(0), u(1)};  // assert control influance
  };

  auto k1 = dxdt(x₀);
  auto k2 = dxdt(x₀ + (dt / 2.0) * k1);
  auto k3 = dxdt(x₀ + (dt / 2.0) * k2);
  auto k4 = dxdt(x₀ + dt * k3);

  return x₀ + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}

}  // namespace rob
