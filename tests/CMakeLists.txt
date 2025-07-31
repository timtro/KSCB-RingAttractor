
find_package(Catch2 REQUIRED)
find_package(TBB REQUIRED COMPONENTS tbb)

macro(create_test target)
  add_test(NAME ${target} COMMAND $<TARGET_FILE:${target}>)
endmacro()

set(testname utest-ringlib)
add_executable(${testname} ${testname}.cpp)
target_link_libraries(${testname} Catch2::Catch2WithMain tbb ringlib)
target_include_directories(${testname} PUBLIC
    "${CMAKE_SOURCE_DIR}/include"
    # "${CMAKE_SOURCE_DIR}/tests/include"
)
create_test(${testname})
