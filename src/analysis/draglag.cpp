#define IMGUI_HAS_DOCK

#include <glad/glad.h>
//
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <implot.h>
#include <stdio.h>

#include <Eigen/Dense>
#include <string>
#include <vector>

#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "ringlib/RingAttractor.hpp"

constexpr double STEP_SIZE = 0.05;
constexpr size_t RING_SIZE = 18;
constexpr double γ = 8.0;
constexpr double κ = 7.0;
constexpr double ν = 0.5;
constexpr double network_coupling_constant = 6;

constexpr double π = std::numbers::pi;

using RingAttractor = ringlib::FeleRingAttractor<RING_SIZE>;

void update_simulation(RingAttractor &attractor, Eigen::VectorXd &input, double &θ_in) {
  input = ringlib::von_mises_input_single<RING_SIZE>(κ, θ_in, γ);
  attractor.update(input, STEP_SIZE);
  θ_in = std::fmod(θ_in + STEP_SIZE / 50, 2.0 * π) - π;
}

void render_ring_plot(const Eigen::VectorXd &neurons) {
  if (neurons.size() == 0) {
    return;
  }

  ImGui::Begin("Neuron Activations");

  if (ImPlot::BeginPlot("Ring Plot", ImVec2(-1, -1),
                        ImPlotFlags_Equal | ImPlotFlags_NoLegend)) {
    ImPlot::SetupAxes(nullptr, nullptr, ImPlotAxisFlags_NoDecorations,
                      ImPlotAxisFlags_NoDecorations);
    ImPlot::SetupAxesLimits(-1.5, 1.5, -1.5, 1.5, ImGuiCond_Once);

    const auto n_neurons = neurons.size();
    std::vector<double> x_pos(RING_SIZE), y_pos(RING_SIZE);
    for (size_t i = 0; i < RING_SIZE; ++i) {
      double angle = ringlib::angle_of<RING_SIZE>(i);
      x_pos[i] = cos(angle);
      y_pos[i] = sin(angle);
    }

    auto normalized_values = neurons;
    double min_val = neurons.minCoeff();
    double max_val = neurons.maxCoeff();
    if (max_val > min_val) {
      normalized_values = ((neurons.array() - min_val) / (max_val - min_val)).sqrt();
    } else {
      normalized_values = Eigen::VectorXd::Zero(n_neurons);
    }

    ImPlot::PushColormap(ImPlotColormap_Viridis);
    for (int i = 0; i < n_neurons; ++i) {
      ImVec4 color = ImPlot::SampleColormap(static_cast<float>(normalized_values(i)));
      ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 8, color, IMPLOT_AUTO, color);
      ImPlot::PlotScatter(("##" + std::to_string(i)).c_str(), &x_pos[i], &y_pos[i], 1);
    }
    ImPlot::PopColormap();

    ImPlot::EndPlot();
  }

  ImGui::End();
}

void render_time_series(const std::vector<Eigen::VectorXd> &history) {
  if (history.empty())
    return;

  size_t n_steps = history.size();
  size_t n_neurons = history[0].size();

  ImGui::Begin("Time Series");

  if (ImPlot::BeginPlot("Neuron Activation Over Time", ImVec2(-1, -1))) {
    for (size_t i = 0; i < n_neurons; ++i) {
      std::vector<double> x(n_steps), y(n_steps);
      for (size_t t = 0; t < n_steps; ++t) {
        x[t] = t;
        y[t] = history[t][i];
      }
      ImPlot::PlotLine(std::to_string(i).c_str(), x.data(), y.data(), n_steps);
    }
    ImPlot::EndPlot();
  }

  ImGui::End();
}

static void glfw_error_callback(int error, const char *description) {
  fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

int main() {
  // Setup window
  glfwSetErrorCallback(glfw_error_callback);
  if (!glfwInit())
    return 1;

  // GL 3.0 + GLSL 130
  const char *glsl_version = "#version 130";
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
  glfwWindowHint(GLFW_MAXIMIZED, GLFW_TRUE);

  GLFWmonitor *monitor = glfwGetPrimaryMonitor();
  const GLFWvidmode *mode = glfwGetVideoMode(monitor);

  // Create window with graphics context
  GLFWwindow *window =
      glfwCreateWindow(mode->width, mode->height, "DragLag Analysis", monitor, NULL);
  if (window == NULL)
    return 1;
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);  // Enable vsync
  // glfwMaximizeWindow(window); // not working

  // Initialize OpenGL loader
  if (gladLoadGL() == 0) {
    fprintf(stderr, "Failed to initialize OpenGL loader!\n");
    return 1;
  }

  // Setup Dear ImGui context
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImPlot::CreateContext();
  ImGuiIO &io = ImGui::GetIO();
  (void)io;
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls
  io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;  // Enable Docking
  io.ConfigFlags |=
      ImGuiConfigFlags_ViewportsEnable;  // Enable Multi-Viewport / Platform Windows

  // Setup DPI scaling
  float scale_factor = 1.6f;  // Your preferred scale factor
  io.FontGlobalScale = scale_factor;

  // Setup Dear ImGui style
  ImGui::StyleColorsDark();

  // Setup Platform/Renderer backends
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init(glsl_version);

  RingAttractor attractor(ν, network_coupling_constant);
  Eigen::VectorXd input = Eigen::VectorXd::Zero(RING_SIZE);
  double θ_in = 0.0;

  std::vector<Eigen::VectorXd> history;
  history.reserve(1000);

  // Main loop
  while (!glfwWindowShouldClose(window)) {
    // Poll and handle events (inputs, window resize, etc.)
    glfwPollEvents();

    // Update simulation
    update_simulation(attractor, input, θ_in);
    history.push_back(attractor.state());
    if (history.size() > 1000) {
      history.erase(history.begin());
    }

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // Enable docking over the main viewport
    ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport());

    // Render plots (now they handle their own windows)
    render_ring_plot(attractor.state());
    render_time_series(history);

    ImGui::SetNextWindowPos(ImVec2(0, 450), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(400, 270), ImGuiCond_FirstUseEver);
    ImGui::Begin("State Display");
    ImGui::Text("Neuron States:");
    for (size_t i = 0; i < attractor.state().size(); ++i) {
      ImGui::Text("Neuron %zu: %.4f", i, attractor.state()[i]);
    }
    ImGui::End();

    // Rendering
    ImGui::Render();
    int display_w, display_h;
    glfwGetFramebufferSize(window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glClearColor(0.45f, 0.55f, 0.60f, 1.00f);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    // Update and Render additional Platform Windows
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
      GLFWwindow *backup_current_context = glfwGetCurrentContext();
      ImGui::UpdatePlatformWindows();
      ImGui::RenderPlatformWindowsDefault();
      glfwMakeContextCurrent(backup_current_context);
    }

    glfwSwapBuffers(window);
  }

  // Cleanup
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImPlot::DestroyContext();
  ImGui::DestroyContext();

  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}
