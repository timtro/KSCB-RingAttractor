#define IMGUI_HAS_DOCK

#include <glad/glad.h>
//
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <implot.h>
#include <spdlog/spdlog.h>
#include <stdio.h>

#include <Eigen/Dense>
#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "ringlib/RingAttractor.hpp"

constexpr double STEP_SIZE = 0.05;
constexpr size_t RING_SIZE = 18;
constexpr double π = std::numbers::pi;
using RingAttractor = ringlib::FeleRingAttractor<RING_SIZE>;

struct Parameters {
  float γ = 2.0f;  // Input gain
  float κ = 20.0f;  // Von Mises concentration
  float ν = 0.5f;  // Kernel parameter
  // float network_coupling_constant = 3.f;  // Network coupling strength
  float network_coupling_constant =
      2.6f;  // Network coupling strength
             // 2.4 - unstable, 2.5 - marginally stable, 2.6 - stable
  float input_speed = 2.0f;  // Speed of input rotation
  bool use_input = true;  // Toggle between von Mises input and zero input
};

void update_simulation(RingAttractor &attractor,
                       Eigen::VectorXd &input,
                       double &θ_in,
                       const Parameters &params) {
  if (params.use_input) {
    input = ringlib::von_mises_input_single<RING_SIZE>(
        static_cast<double>(params.κ), θ_in, static_cast<double>(params.γ));
    θ_in = wrap_angle(θ_in + STEP_SIZE * static_cast<double>(params.input_speed));
  } else {
    input = Eigen::VectorXd::Zero(RING_SIZE);
    // Don't update θ_in when input is disabled
  }
  attractor.update(input, STEP_SIZE);
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
    for (size_t i = 0; i < static_cast<size_t>(n_neurons); ++i) {
      ImVec4 color = ImPlot::SampleColormap(
          static_cast<float>(normalized_values(static_cast<int>(i))));
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
  size_t n_neurons = static_cast<size_t>(history[0].size());

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
        x[t] = static_cast<double>(t);
        y[t] = history[t][static_cast<int>(i)];
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
      ImPlot::PlotLine(std::to_string(i).c_str(), x.data(), y.data(),
                       static_cast<int>(n_steps));
    }
    ImPlot::EndPlot();
  }

  ImGui::End();
}

// Plot RMSE between ring attractor heading and input angle over time
void render_mse_plot(const std::vector<double> &mse_history) {
  if (mse_history.empty())
    return;

  size_t n_steps = mse_history.size();

  ImGui::Begin("Heading RMSE");

  if (ImPlot::BeginPlot("Root Mean Square Error: Heading vs Input Angle",
                        ImVec2(-1, -1))) {
    // Set up dynamic X-axis that follows the data
    double x_min = 0.0;
    double x_max = static_cast<double>(n_steps - 1);

    // Always show at least a reasonable window
    if (x_max < 100.0) {
      x_max = 100.0;
    }

    ImPlot::SetupAxisLimits(ImAxis_X1, x_min, x_max, ImGuiCond_Always);
    ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, ImPlotCond_Once);  // RMSE is always >= 0

    std::vector<double> x(n_steps);
    for (size_t t = 0; t < n_steps; ++t) {
      x[t] = static_cast<double>(t);
    }

    // Plot RMSE with a distinctive color
    ImPlot::SetNextLineStyle(ImVec4(1.0f, 0.4f, 0.2f, 1.0f), 2.0f);  // Orange line
    ImPlot::PlotLine("RMSE (radians)", x.data(), mse_history.data(),
                     static_cast<int>(n_steps));

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
      heatmap_data[i] =
          static_cast<float>((input[static_cast<int>(i)] - input_min) / input_range);
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
      heatmap_data[RING_SIZE + i] = static_cast<float>(
          (neurons[static_cast<int>(i)] - neurons_min) / neurons_range);
    } else {
      heatmap_data[RING_SIZE + i] = 0.0f;
    }
  }

  // Use [0, 1] range for colormap since both rows are now normalized
  double scale_min = 0.0;
  double scale_max = 1.0;

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
                          bool &ring_attractor_needs_reconstruction,
                          std::vector<Eigen::VectorXd> &neuron_history,
                          std::vector<double> &mse_history,
                          bool &is_paused,
                          double &sum_squared_errors,
                          size_t &error_count) {
  ImGui::Begin("Control Panel");

  ImGui::Text("Tuning Parameters");
  ImGui::Separator();

  ImGui::Checkbox("Oscillating Input", &params.use_input);
  ImGui::Separator();

  bool gamma_changed =
      ImGui::DragFloat("Input Gain (γ)", &params.γ, 0.1f, 0.1f, 20.0f, "%.2f");
  bool kappa_changed =
      ImGui::DragFloat("Von Mises κ", &params.κ, 0.1f, 0.1f, 20.0f, "%.2f");
  bool nu_changed = ImGui::DragFloat("Kernel ν", &params.ν, 0.001f, 0.01f, 1.0f, "%.3f");
  bool coupling_changed = ImGui::DragFloat("Coupling", &params.network_coupling_constant,
                                           0.1f, 0.1f, 20.0f, "%.2f");
  bool speed_changed =
      ImGui::DragFloat("Input Speed", &params.input_speed, 0.1f, 0.0f, 10.0f, "%.2f");

  // If ν or coupling parameter change, the ring attractor needs reconstruction.
  if (nu_changed || coupling_changed) {
    ring_attractor_needs_reconstruction = true;
  }

  ImGui::Separator();
  ImGui::Text("Simulation Controls");

  if (is_paused) {
    if (ImGui::Button("Resume (Space)")) {
      is_paused = false;
    }
  } else {
    if (ImGui::Button("Pause (Space)")) {
      is_paused = true;
    }
  }

  ImGui::Separator();
  ImGui::Text("Network Controls");

  if (ImGui::Button("Zero Network")) {
    attractor.neurons.setZero();
    neuron_history.clear();  // Clear neuron history
    mse_history.clear();  // Clear RMSE history
    sum_squared_errors = 0.0;  // Reset RMSE accumulation
    error_count = 0;
  }

  if (ImGui::Button("Reset Parameters")) {
    params = Parameters{};
    ring_attractor_needs_reconstruction = true;
    neuron_history.clear();
    mse_history.clear();
    sum_squared_errors = 0.0;  // Reset RMSE accumulation
    error_count = 0;
  }

  ImGui::End();
}

static void glfw_error_callback(int error, const char *description) {
  fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

auto main() -> int {
  spdlog::set_level(spdlog::level::info);
  spdlog::info("Starting DragLag Analysis application");

  // Log current working directory for debugging font path issues
  std::filesystem::path cwd = std::filesystem::current_path();
  spdlog::info("Current working directory: {}", cwd.string());

  std::filesystem::path font_path = "../assets/DejaVuSans.ttf";
  bool font_exists = std::filesystem::exists(font_path);
  spdlog::info("Font file exists at {}: {}", font_path.string(), font_exists);

  if (font_exists) {
    auto font_size = std::filesystem::file_size(font_path);
    spdlog::info("Font file size: {} bytes", font_size);
  }

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

  // DPI scaling
  float gui_scale_factor = 1.6f;
  io.FontGlobalScale = gui_scale_factor;

  // Load font with Greek character support
  spdlog::info("Attempting to load font from: ../assets/DejaVuSans.ttf");

  ImFontConfig config;
  config.MergeMode = false;

  // Load the main font with default Latin characters
  // Path is relative to executable location (build/) so run from there.
  ImFont *font = io.Fonts->AddFontFromFileTTF("../assets/DejaVuSans.ttf",
                                              9.0f * gui_scale_factor, &config);

  if (font == nullptr) {
    spdlog::error("Failed to load main font from ../assets/DejaVuSans.ttf");
    spdlog::info("Falling back to default font");
  } else {
    spdlog::info("Successfully loaded main font");
  }

  // Add Greek character range to the same font
  config.MergeMode = true;  // Merge with the previous font
  static const ImWchar greek_ranges[] = {
      0x0370,
      0x03FF,  // Greek and Coptic block
      0,
  };
  ImFont *greek_font = io.Fonts->AddFontFromFileTTF(
      "../assets/DejaVuSans.ttf", 9.0f * gui_scale_factor, &config, greek_ranges);

  if (greek_font == nullptr) {
    spdlog::error("Failed to load Greek character range from ../assets/DejaVuSans.ttf");
  } else {
    spdlog::info("Successfully loaded Greek character range");
  }

  // Build font atlas
  bool font_build_success = io.Fonts->Build();
  spdlog::info("Font atlas build result: {}", font_build_success ? "success" : "failed");

  if (io.Fonts->Fonts.Size == 0) {
    spdlog::error("No fonts available after loading attempt");
  } else {
    spdlog::info("Total fonts loaded: {}", io.Fonts->Fonts.Size);
  }

  // Setup Dear ImGui style
  ImGui::StyleColorsDark();

  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init(glsl_version);

  // Setup simulation and GUI state
  Parameters params;
  RingAttractor attractor(static_cast<double>(params.ν),
                          static_cast<double>(params.network_coupling_constant));
  Eigen::VectorXd input = Eigen::VectorXd::Zero(RING_SIZE);
  double θ_in = 0.0;
  bool needs_reconstruction = false;
  bool is_paused = false;  // Simulation pause state

  std::vector<Eigen::VectorXd> history;
  std::vector<double> mse_history;
  constexpr size_t MAX_HISTORY = 1000;
  history.reserve(MAX_HISTORY);
  mse_history.reserve(MAX_HISTORY);

  // RMSE calculation variables
  double sum_squared_errors = 0.0;
  size_t error_count = 0;

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();

    // Reconstruct attractor if it's construction parameters are changed
    if (needs_reconstruction) {
      attractor = RingAttractor(static_cast<double>(params.ν),
                                static_cast<double>(params.network_coupling_constant));
      needs_reconstruction = false;
    }

    if (!is_paused) {
      update_simulation(attractor, input, θ_in, params);
      history.push_back(attractor.state());
      if (history.size() > MAX_HISTORY) {
        history.erase(history.begin());
      }

      // Calculate and accumulate squared angular error for RMSE
      auto angle_error = [](double angle1, double angle2) {
        double diff = angle1 - angle2;
        // Wrap difference to [-π, π]
        diff = wrap_angle(diff);
        return diff;  // Return signed difference
      };

      double heading = attractor.heading();
      double error = angle_error(heading, θ_in);
      double squared_error = error * error;

      // Accumulate for RMSE calculation
      sum_squared_errors += squared_error;
      error_count++;

      // Calculate current RMSE and store for plotting
      double current_rmse = std::sqrt(sum_squared_errors / error_count);
      mse_history.push_back(current_rmse);

      if (mse_history.size() > MAX_HISTORY) {
        mse_history.erase(mse_history.begin());
        // Don't reset accumulation - we want running RMSE over all time
      }
    }

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // Space-bar to pause
    static bool space_pressed = false;
    bool space_key_down = ImGui::IsKeyDown(ImGuiKey_Space);

    if (space_key_down && !space_pressed) {
      is_paused = !is_paused;
      space_pressed = true;
    } else if (!space_key_down) {
      space_pressed = false;
    }

    // Enable docking over the main viewport
    ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport());

    render_control_panel(params, attractor, needs_reconstruction, history, mse_history,
                         is_paused, sum_squared_errors, error_count);
    render_ring_plot(attractor.state());
    render_time_series(history);
    render_mse_plot(mse_history);
    render_heatmap(attractor.state(), input);

    ImGui::SetNextWindowPos(ImVec2(0, 450), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(400, 270), ImGuiCond_FirstUseEver);
    ImGui::Begin("State Display");
    ImGui::Text("Neuron States:");
    for (size_t i = 0; i < static_cast<size_t>(attractor.state().size()); ++i) {
      ImGui::Text("Neuron %zu: %.4f", i, attractor.state()[static_cast<int>(i)]);
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
