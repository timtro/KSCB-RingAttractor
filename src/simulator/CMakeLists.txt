find_package(ZeroMQ REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(cppzmq REQUIRED)

add_executable(simulator main.cpp)
target_include_directories(simulator PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(simulator
    ringlib
    libzmq-static
    cppzmq
    nlohmann_json::nlohmann_json
)

