find_package(ZeroMQ REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(cppzmq REQUIRED)

find_path(IMGUI_BACKENDS_DIR
          NAMES imgui_impl_glfw.cpp
          HINTS ${imgui_PACKAGE_FOLDER_DEBUG} ${imgui_PACKAGE_FOLDER_RELEASE}
          PATH_SUFFIXES res/bindings)

set(DRAGLAG_SOURCES
    main.cpp
    # ${IMGUI_BACKENDS_DIR}/imgui_impl_glfw.cpp
    # ${IMGUI_BACKENDS_DIR}/imgui_impl_opengl3.cpp
)

add_executable(simulator main.cpp)
target_include_directories(simulator PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(simulator
    ringlib
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    # imgui::imgui
    # implot::implot
    # glfw
    # glad::glad
)

