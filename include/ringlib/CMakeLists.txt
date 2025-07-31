find_package(Eigen3 REQUIRED)

add_library(ringlib INTERFACE)

target_include_directories(ringlib INTERFACE
    Eigen3::Eigen
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(ringlib INTERFACE Eigen3::Eigen)
