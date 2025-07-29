
find_package(Catch2 REQUIRED)

macro (create_test target)
  add_test (NAME ${target} COMMAND $<TARGET_FILE:${target}>)
endmacro (create_test)

# Function to help faciliate setting up unit tests
function(MakeTests)
    # Parse arguments
    set(options)
    set(oneValueArgs)
    set(multiValueArgs PUBLIC_TARGETS PRIVATE_TARGETS PUBLIC_DEFINITIONS PRIVATE_DEFINITIONS PRIVATE_INCLUDE_DIRECTORIES)
    cmake_parse_arguments(MAKE_TESTS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    get_filename_component(library_MODULE ${CMAKE_CURRENT_SOURCE_DIR} NAME)
    string(REPLACE " " "_" library_MODULE ${library_MODULE})

    #if( NOT ${PROJECT_NAME}_BUILD_UNIT_TESTS OR NOT IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/test")
    #    return()
    #endif()

    #find_package(Catch2 REQUIRED)
    #enable_testing()

    file(GLOB files "*.cpp")

    set(TESTCASE_PREFIX "${PROJECT_NAME}-${library_MODULE}")

    message("\n    UNIT TESTS:")
    foreach(file ${files})

        get_filename_component(file_basename ${file} NAME_WE)

        set(UNIT_EXE_NAME  ${PROJECT_NAME}-${library_MODULE}-${file_basename} )
        set(UNIT_TEST_NAME ${PROJECT_NAME}-${library_MODULE}-${file_basename} )

        add_executable( ${UNIT_EXE_NAME} ${file} )

        message("        ${UNIT_EXE_NAME}")
        target_link_libraries( ${UNIT_EXE_NAME} PUBLIC
                                                    ${UNIT_TEST_LINK_TARGETS}
                                                    ${MAKE_TESTS_PRIVATE_TARGETS}
                                                    ${MAKE_TESTS_PUBLIC_TARGETS})

        target_include_directories(${UNIT_EXE_NAME} PRIVATE
                                        ${MAKE_TESTS_PRIVATE_INCLUDE_DIRECTORIES})
        target_compile_definitions(${UNIT_EXE_NAME} PRIVATE
                                        ${MAKE_TESTS_PRIVATE_DEFINITIONS})

        target_compile_definitions(${UNIT_EXE_NAME} PRIVATE
                                        ${MAKE_TESTS_PUBLIC_DEFINITIONS})

        add_test(NAME ${UNIT_TEST_NAME}
                 COMMAND ${UNIT_EXE_NAME}
        )

        list(APPEND unitTestsList ${UNIT_EXE_NAME})

        set_target_properties(${UNIT_EXE_NAME}
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/test
        )
    endforeach()
    message("*****************************************************")


endfunction()
