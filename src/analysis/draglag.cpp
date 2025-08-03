#define IMGUI_HAS_DOCK

#include <glad/glad.h>
//
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <implot.h>
#include <stdio.h>

#include <Eigen/Dense>
#include <algorithm>
#include <string>
#include <vector>

#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "ringlib/RingAttractor.hpp"

constexpr double STEP_SIZE = 0.05;
constexpr size_t RING_SIZE = 18;
constexpr double π = std::numbers::pi;

// Tunable parameters (with default values)
struct Parameters {
  float γ = 8.0f;  // Input gain
  float κ = 8.0f;  // Von Mises concentration
  float ν = 0.10f;  // Kernel parameter
  float network_coupling_constant = 6.0f;  // Network coupling strength
  float input_speed = 2.0f;  // Speed of input rotation
};

using RingAttractor = ringlib::FeleRingAttractor<RING_SIZE>;

void update_simulation(RingAttractor &attractor,
                       Eigen::VectorXd &input,
                       double &θ_in,
                       const Parameters &params) {
  input = ringlib::von_mises_input_single<RING_SIZE>(static_cast<double>(params.κ), θ_in,
                                                     static_cast<double>(params.γ));
  attractor.update(input, STEP_SIZE);
  θ_in = wrap_angle(θ_in + STEP_SIZE * static_cast<double>(params.input_speed));
}

// Ring plot shows the ring of neurons as points and colours them by activity level.
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

// Plot traces of neuonal activation. Traces are coloured based on their angular position
// on the color wheel: red is 0°, cyan is 180°, etc.
void render_time_series(const std::vector<Eigen::VectorXd> &history) {
  if (history.empty())
    return;

  size_t n_steps = history.size();
  size_t n_neurons = history[0].size();

  ImGui::Begin("Time Series");

  if (ImPlot::BeginPlot("Neuron Activation Over Time", ImVec2(-1, -1))) {
    // Set up dynamic X-axis that follows the data
    double x_min = 0.0;
    double x_max = static_cast<double>(n_steps - 1);

    // Always show at least a reasonable window
    if (x_max < 100.0) {
      x_max = 100.0;
    }

    ImPlot::SetupAxisLimits(ImAxis_X1, x_min, x_max, ImGuiCond_Always);

    for (size_t i = 0; i < n_neurons; ++i) {
      std::vector<double> x(n_steps), y(n_steps);
      for (size_t t = 0; t < n_steps; ++t) {
        x[t] = t;
        y[t] = history[t][i];
      }

      // Color each neuron based on its angular position on the ring
      double angle = ringlib::angle_of<RING_SIZE>(i);  // angle in radians [0, 2π)
      double hue = angle / (2.0 * π);  // normalize to [0, 1] for hue

      // Convert HSV to RGB (with full saturation and value)
      auto hsv_to_rgb = [](double h, double s, double v) -> ImVec4 {
        double c = v * s;
        double x_comp = c * (1.0 - std::abs(std::fmod(h * 6.0, 2.0) - 1.0));
        double m = v - c;

        double r, g, b;
        if (h < 1.0 / 6.0) {
          r = c;
          g = x_comp;
          b = 0;
        } else if (h < 2.0 / 6.0) {
          r = x_comp;
          g = c;
          b = 0;
        } else if (h < 3.0 / 6.0) {
          r = 0;
          g = c;
          b = x_comp;
        } else if (h < 4.0 / 6.0) {
          r = 0;
          g = x_comp;
          b = c;
        } else if (h < 5.0 / 6.0) {
          r = x_comp;
          g = 0;
          b = c;
        } else {
          r = c;
          g = 0;
          b = x_comp;
        }

        return ImVec4(static_cast<float>(r + m), static_cast<float>(g + m),
                      static_cast<float>(b + m), 1.0f);
      };

      ImVec4 color = hsv_to_rgb(hue, 0.8, 0.9);  // 80% saturation, 90% brightness
      ImPlot::SetNextLineStyle(color);
      ImPlot::PlotLine(std::to_string(i).c_str(), x.data(), y.data(), n_steps);
    }
    ImPlot::EndPlot();
  }

  ImGui::End();
}

// Neuron activity and input signals are compared as quantitatively annotated heatmaps.
void render_heatmap(const Eigen::VectorXd &neurons, const Eigen::VectorXd &input) {
  if (neurons.size() == 0 || input.size() == 0) {
    return;
  }

  ImGui::Begin("Neural Activity Heatmap");

  // Create heatmap data: 2 rows x RING_SIZE columns
  // Without ColMajor flag, data should be laid out row by row
  float heatmap_data[2 * RING_SIZE];

  // Normalize each row independently for better visibility
  //
  // Row 0: von Mises input vector (normalized)
  double input_min = input.minCoeff();
  double input_max = input.maxCoeff();
  double input_range = input_max - input_min;
  for (size_t i = 0; i < RING_SIZE; ++i) {
    if (input_range > 0) {
      heatmap_data[i] = static_cast<float>((input[i] - input_min) / input_range);
    } else {
      heatmap_data[i] = 0.0f;
    }
  }

  // Row 1: ring attractor state (normalized)
  double neurons_min = neurons.minCoeff();
  double neurons_max = neurons.maxCoeff();
  double neurons_range = neurons_max - neurons_min;
  for (size_t i = 0; i < RING_SIZE; ++i) {
    if (neurons_range > 0) {
      heatmap_data[RING_SIZE + i] =
          static_cast<float>((neurons[i] - neurons_min) / neurons_range);
    } else {
      heatmap_data[RING_SIZE + i] = 0.0f;
    }
  }

  // Use [0, 1] range for colormap since both rows are now normalized
  float scale_min = 0.0f;
  float scale_max = 1.0f;

  static const char *row_labels[] = {"Input", "Neurons"};

  static std::vector<std::string> col_label_strings;
  static std::vector<const char *> col_labels;
  static bool labels_initialized = false;

  if (!labels_initialized) {
    col_label_strings.reserve(RING_SIZE);
    col_labels.reserve(RING_SIZE);
    for (size_t i = 0; i < RING_SIZE; ++i) {
      col_label_strings.push_back(std::to_string(i));
      col_labels.push_back(col_label_strings.back().c_str());
    }
    labels_initialized = true;
  }

  if (ImPlot::BeginPlot("Neural Activity Heatmap", ImVec2(-1, -1),
                        ImPlotFlags_NoLegend | ImPlotFlags_NoMouseText)) {

    // Setup axes with appropriate flags
    ImPlotAxisFlags axes_flags =
        ImPlotAxisFlags_Lock | ImPlotAxisFlags_NoGridLines | ImPlotAxisFlags_NoTickMarks;
    ImPlot::SetupAxes(nullptr, nullptr, axes_flags, axes_flags);

    // Setup axis ticks and labels
    ImPlot::SetupAxisTicks(ImAxis_X1, 0 + 1.0 / (2 * RING_SIZE),
                           1 - 1.0 / (2 * RING_SIZE), RING_SIZE, col_labels.data());
    ImPlot::SetupAxisTicks(ImAxis_Y1, 1 - 1.0 / 4.0, 0 + 1.0 / 4.0, 2, row_labels);

    // Use Viridis colormap
    ImPlot::PushColormap(ImPlotColormap_Viridis);

    // Plot heatmap
    ImPlot::PlotHeatmap("##heatmap", heatmap_data, 2, RING_SIZE, scale_min, scale_max,
                        "%.3f", ImPlotPoint(0, 0), ImPlotPoint(1, 1));

    ImPlot::PopColormap();
    ImPlot::EndPlot();
  }

  // Add colormap scale on the side
  ImGui::SameLine();
  ImPlot::ColormapScale("##HeatScale", scale_min, scale_max, ImVec2(60, -1));

  ImGui::End();
}

// Control panel for tuning parameters and network controls
void render_control_panel(Parameters &params,
                          RingAttractor &attractor,
                          bool &needs_reconstruction) {
  ImGui::Begin("Control Panel");

  ImGui::Text("Tuning Parameters");
  ImGui::Separator();

  // Parameter drag inputs (scrollable/draggable textboxes)
  bool gamma_changed =
      ImGui::DragFloat("Input Gain (γ)", &params.γ, 0.1f, 0.1f, 20.0f, "%.2f");
  bool kappa_changed =
      ImGui::DragFloat("Von Mises κ", &params.κ, 0.1f, 0.1f, 20.0f, "%.2f");
  bool nu_changed = ImGui::DragFloat("Kernel ν", &params.ν, 0.001f, 0.01f, 1.0f, "%.3f");
  bool coupling_changed = ImGui::DragFloat("Coupling", &params.network_coupling_constant,
                                           0.1f, 0.1f, 20.0f, "%.2f");
  bool speed_changed =
      ImGui::DragFloat("Input Speed", &params.input_speed, 0.1f, 0.0f, 10.0f, "%.2f");

  // Mark for reconstruction if kernel parameter or coupling changed
  if (nu_changed || coupling_changed) {
    needs_reconstruction = true;
  }

  ImGui::Separator();
  ImGui::Text("Network Controls");

  // Zero out network button
  if (ImGui::Button("Zero Network")) {
    attractor.neurons.setZero();
  }

  // Reset parameters button
  if (ImGui::Button("Reset Parameters")) {
    params = Parameters{};  // Reset to defaults
    needs_reconstruction = true;
  }

  ImGui::End();
}

static void glfw_error_callback(int error, const char *description) {
  fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

auto main() -> int {
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

  // Create window with graphics context, using monitor's resolution for fullscreen. This
  // is a bit of a hack since the glfwMaximizeWindow() function doesn't work on
  // Wayland/Hyprland.
  GLFWwindow *window =
      glfwCreateWindow(mode->width, mode->height, "DragLag Analysis", monitor, NULL);
  if (window == NULL)
    return 1;
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);  // Enable vsync for smooth rendering
  // glfwMaximizeWindow(window); // not working

  // Initialize OpenGL loader
  if (gladLoadGL() == 0) {
    fprintf(stderr, "Failed to initialize OpenGL loader!\n");
    return 1;
  }

  // ImGui and ImPlot constext setup.
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
  float scale_factor = 1.6f;
  io.FontGlobalScale = scale_factor;

  // Setup Dear ImGui style
  ImGui::StyleColorsDark();

  // Setup Platform/Renderer backends
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init(glsl_version);

  Parameters params;
  RingAttractor attractor(static_cast<double>(params.ν),
                          static_cast<double>(params.network_coupling_constant));
  Eigen::VectorXd input = Eigen::VectorXd::Zero(RING_SIZE);
  double θ_in = 0.0;
  bool needs_reconstruction = false;

  std::vector<Eigen::VectorXd> history;
  constexpr size_t MAX_HISTORY = 1000;
  history.reserve(MAX_HISTORY);

  // Main loop
  while (!glfwWindowShouldClose(window)) {
    // Poll and handle events (inputs, window resize, etc.)
    glfwPollEvents();

    // Reconstruct attractor if it's construction parameters are changed
    if (needs_reconstruction) {
      attractor = RingAttractor(static_cast<double>(params.ν),
                                static_cast<double>(params.network_coupling_constant));
      needs_reconstruction = false;
    }

    // Update simulation
    update_simulation(attractor, input, θ_in, params);
    history.push_back(attractor.state());
    if (history.size() > MAX_HISTORY) {
      history.erase(history.begin());
    }

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // Enable docking over the main viewport
    ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport());

    render_control_panel(params, attractor, needs_reconstruction);
    render_ring_plot(attractor.state());
    render_time_series(history);
    render_heatmap(attractor.state(), input);

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

    // Force GPU synchronization (potential fix for visual artifacts)
    glFinish();

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
