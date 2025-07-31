#include <cmath>
#include <nlohmann/json.hpp>
#include <print>
#include <zmq.hpp>

#include "ringlib/RingAttractor.hpp"

using json = nlohmann::json;

using RobotState = Eigen::Matrix<double, 5, 1>;
using ControlSpace = Eigen::Matrix<double, 2, 1>;

constexpr double STEP_SIZE = 0.01;

std::tuple<double &, double &, double &, double &, double &> by_elements(RobotState &x) {
  return std::tie(x(0), x(1), x(2), x(3), x(4));
}

std::tuple<const double &, const double &, const double &, const double &, const double &>
    by_elements(const RobotState &x) {
  return std::tie(x(0), x(1), x(2), x(3), x(4));
}

auto rk4_step(const RobotState &x₀, const ControlSpace &u, double dt) -> RobotState {
  auto dxdt = [&u](const RobotState &state) -> RobotState {
    const auto &[x, y, θ, v, ω] = by_elements(state);
    return {
        v * std::cos(θ),  // dx/dt = v * cos(θ)
        v * std::sin(θ),  // dy/dt = v * sin(θ)
        ω,  // dθ/dt = ω
        u(0),  // dv/dt (example: no linear acceleration)
        u(1)  // dω/dt (example: no angular acceleration)
    };
  };

  auto k1 = dxdt(x₀);
  auto k2 = dxdt(x₀ + (dt / 2.0) * k1);
  auto k3 = dxdt(x₀ + (dt / 2.0) * k2);
  auto k4 = dxdt(x₀ + dt * k3);

  return x₀ + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}

struct RobotModel {
  RobotState state;

  auto x() const -> double { return state(0); };
  auto y() const -> double { return state(1); };
  auto θ() const -> double { return state(2); };
  auto v() const -> double { return state(3); };
  auto ω() const -> double { return state(4); };

  void update(ControlSpace u) { state = rk4_step(state, u, STEP_SIZE); };
};

auto to_json(const RobotModel &robot) -> json {
  return json{{"x", robot.x()},
              {"y", robot.y()},
              {"θ", robot.θ()},
              {"v", robot.v()},
              {"ω", robot.ω()}};
}

template <size_t N>
auto to_json(const ringlib::FeleRingAttractor<N> &attractor) -> json {
  return json{{"neurons", attractor.state().transpose().eval()}};
}

int main() {

  ringlib::FeleRingAttractor<18> ring_attractor(0.15);

  zmq::context_t context(1);
  zmq::socket_t responder(context, ZMQ_REP);
  responder.bind("ipc:///tmp/zmq-sim.sock");
  std::println("Simulator server started on ipc:///tmp/zmq-sim.sock");

  ring_attractor.update({1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 0.01);

  while (true) {
    zmq::message_t request;
    auto recv_result = responder.recv(request, zmq::recv_flags::none);
    if (!recv_result || *recv_result == 0) {
      continue;
    }

    std::string req_str(static_cast<char *>(request.data()), request.size());
    json req_json;
    try {
      req_json = json::parse(req_str);
    } catch (...) {
      responder.send(zmq::buffer("{\"error\":\"bad request\"}"), zmq::send_flags::none);
      continue;
    }

    ring_attractor.update(decltype(ring_attractor.neurons)::Zero(), 0.06);

    std::string type = req_json.value("type", "get_state");
    if (type == "get_state") {
      std::string msg_str = to_json(ring_attractor).dump();
      responder.send(zmq::buffer(msg_str), zmq::send_flags::none);
    } else {
      responder.send(zmq::buffer("{\"error\":\"unknown request type\"}"),
                     zmq::send_flags::none);
    }
  }
}
