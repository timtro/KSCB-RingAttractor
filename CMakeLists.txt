# vim: fdm=marker:fdc=2:fmr={{{,}}}
cmake_minimum_required(VERSION 3.25)

project(canopy LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(${PROJECT_NAME}_BUILD_UNIT_TESTS
  "Build the unit tests for this library"
  ${PROJECT_IS_TOP_LEVEL}
)
option(${PROJECT_NAME}_ENABLE_WARNINGS
  "Enable Strict Warnings"
  ${PROJECT_IS_TOP_LEVEL}
)
option(
  ${PROJECT_NAME}_WARNINGS_AS_ERRORS
  "Treat compiler warnings as errors"
  ON
)

# Compiler static analysis options ........................................ {{{1
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options($<$<CONFIG:Debug>:-fsanitize=address,undefined>)
  add_link_options($<$<CONFIG:Debug>:-fsanitize=address,undefined>)
  add_compile_options(-fdiagnostics-color=always)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # add_compile_options($<$<CONFIG:Debug>:-fanalyzer>)
    add_compile_options(-fstack-protector-strong)
    add_compile_options($<$<NOT:$<CONFIG:Debug>>:-D_FORTIFY_SOURCE=2>)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options($<$<CONFIG:Debug>:-fanalyze>)
    add_compile_options(-fstack-protector-all)

    # Thread sanitizer (cannot be combined with ASan)
    # add_compile_options($<$<CONFIG:Debug>:-fsanitize=thread>)
    # add_link_options($<$<CONFIG:Debug>:-fsanitize=thread>)

    # Memory sanitizer (Clang-only, requires instrumented libc++)
    # add_compile_options($<$<CONFIG:Debug>:-fsanitize=memory>)
    # add_link_options($<$<CONFIG:Debug>:-fsanitize=memory>)

    # Leak sanitizer (can be combined with ASan)
    add_compile_options($<$<CONFIG:Debug>:-fsanitize=leak>)
    add_link_options($<$<CONFIG:Debug>:-fsanitize=leak>)
  endif()
  add_compile_options($<$<CONFIG:Debug>:-fsanitize-address-use-after-scope>)

  # Control flow integrity (CFI) options (modern compilers)
  # Requires LTO to be enabled
  # add_compile_options($<$<NOT:$<CONFIG:Debug>>:-fsanitize=cfi -flto -fvisibility=hidden>)
  # add_link_options($<$<NOT:$<CONFIG:Debug>>:-fsanitize=cfi -flto>)

  # Integer overflow checking (may need to be selective, can cause performance impact)
  # add_compile_options($<$<CONFIG:Debug>:-fsanitize=integer>)
  # add_link_options($<$<CONFIG:Debug>:-fsanitize=integer>)

  # Sanitizer options that help with debugging
  add_compile_options($<$<CONFIG:Debug>:-fno-omit-frame-pointer>)
  add_compile_options($<$<CONFIG:Debug>:-fno-optimize-sibling-calls>)
endif()
#  ........................................................................ }}}1

# include(cmake/extras.cmake)

# Conan dependencies
find_package(spdlog REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(ZeroMQ REQUIRED)
find_package(cppzmq REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(Catch2 REQUIRED)
find_package(imgui REQUIRED)
find_package(implot REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glad REQUIRED)

add_subdirectory(src)
add_subdirectory(include)

if(${PROJECT_NAME}_BUILD_UNIT_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# Post configuration ...................................................... {{{1
# Copy compile commands JSON to top level for clangd.
if( EXISTS "${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json" )
  execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json
    ${CMAKE_CURRENT_SOURCE_DIR}/compile_commands.json
  )
ENDIF()
#  ........................................................................ }}}1
