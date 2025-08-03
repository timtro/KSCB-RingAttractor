find_path(IMGUI_BACKENDS_DIR
          NAMES imgui_impl_glfw.cpp
          HINTS ${imgui_PACKAGE_FOLDER_DEBUG} ${imgui_PACKAGE_FOLDER_RELEASE}
          PATH_SUFFIXES res/bindings)

set(DRAGLAG_SOURCES
    draglag.cpp
    ${IMGUI_BACKENDS_DIR}/imgui_impl_glfw.cpp
    ${IMGUI_BACKENDS_DIR}/imgui_impl_opengl3.cpp
)

add_executable(draglag ${DRAGLAG_SOURCES})

target_include_directories(draglag PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${IMGUI_BACKENDS_DIR}
)

target_link_libraries(draglag
    ringlib
    imgui::imgui
    implot::implot
    glfw
    glad::glad
)

