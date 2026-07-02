set(INFINI_RT_INCLUDE_DIR "${INFINI_RT_INSTALL_PREFIX}/include")
set(INFINI_RT_LIBRARY_DIR "${INFINI_RT_INSTALL_PREFIX}/lib")

if(NOT EXISTS "${INFINI_RT_LIBRARY_DIR}/libinfinirt.so")
    set(INFINI_RT_LIBRARY_DIR "${INFINI_RT_INSTALL_PREFIX}/lib64")
endif()

if(NOT EXISTS "${INFINI_RT_INCLUDE_DIR}/infini/rt.h")
    message(FATAL_ERROR "The installed public header was not found.")
endif()

if(NOT EXISTS "${INFINI_RT_LIBRARY_DIR}/libinfinirt.so")
    message(FATAL_ERROR "The installed InfiniRT library was not found.")
endif()

set(INFINI_RT_LD_LIBRARY_PATH "${INFINI_RT_LIBRARY_DIR}")

if(INFINI_RT_EXTRA_LIBRARY_PATHS)
    string(REPLACE ":" ";" INFINI_RT_EXTRA_LIBRARY_DIRS
           "${INFINI_RT_EXTRA_LIBRARY_PATHS}")
    foreach(INFINI_RT_EXTRA_LIBRARY_DIR ${INFINI_RT_EXTRA_LIBRARY_DIRS})
        if(EXISTS "${INFINI_RT_EXTRA_LIBRARY_DIR}")
            set(INFINI_RT_LD_LIBRARY_PATH
                "${INFINI_RT_LD_LIBRARY_PATH}:${INFINI_RT_EXTRA_LIBRARY_DIR}")
        endif()
    endforeach()
endif()

get_filename_component(INFINI_RT_CONSUMER_OUTPUT_DIR
                       "${INFINI_RT_CONSUMER_BINARY}" DIRECTORY)
get_filename_component(INFINI_RT_CONSUMER_OUTPUT_NAME
                       "${INFINI_RT_CONSUMER_BINARY}" NAME)
set(INFINI_RT_CONSUMER_PROJECT_DIR
    "${INFINI_RT_CONSUMER_BINARY}.project")
set(INFINI_RT_CONSUMER_BUILD_DIR
    "${INFINI_RT_CONSUMER_BINARY}.build")

file(MAKE_DIRECTORY "${INFINI_RT_CONSUMER_PROJECT_DIR}")
file(WRITE "${INFINI_RT_CONSUMER_PROJECT_DIR}/CMakeLists.txt" [=[
cmake_minimum_required(VERSION 3.18)
project(InfiniRTInstallConsumer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(absl CONFIG REQUIRED)

add_library(infinirt SHARED IMPORTED GLOBAL)
set_target_properties(infinirt PROPERTIES
    IMPORTED_LOCATION "${INFINI_RT_LIBRARY_DIR}/libinfinirt.so")

add_executable(install_consumer "${INFINI_RT_CONSUMER_SOURCE}")
target_include_directories(install_consumer PRIVATE "${INFINI_RT_INCLUDE_DIR}")
target_compile_options(install_consumer PRIVATE -Werror)
target_link_libraries(install_consumer
    PRIVATE
        infinirt
        absl::status
        absl::statusor)
set_target_properties(install_consumer PROPERTIES
    OUTPUT_NAME "${INFINI_RT_CONSUMER_OUTPUT_NAME}"
    RUNTIME_OUTPUT_DIRECTORY "${INFINI_RT_CONSUMER_OUTPUT_DIR}"
    BUILD_RPATH "${INFINI_RT_LIBRARY_DIR}")

if(INFINI_RT_CONSUMER_BACKEND AND
   NOT INFINI_RT_CONSUMER_BACKEND STREQUAL "NONE")
    target_compile_definitions(install_consumer
        PRIVATE INFINI_RT_CONSUMER_BACKEND_${INFINI_RT_CONSUMER_BACKEND}=1)
endif()

if(INFINI_RT_EXTRA_INCLUDE_PATHS)
    string(REPLACE ":" ";" INFINI_RT_EXTRA_INCLUDE_DIRS
           "${INFINI_RT_EXTRA_INCLUDE_PATHS}")
    foreach(INFINI_RT_EXTRA_INCLUDE_DIR IN LISTS INFINI_RT_EXTRA_INCLUDE_DIRS)
        if(EXISTS "${INFINI_RT_EXTRA_INCLUDE_DIR}")
            target_include_directories(install_consumer
                PRIVATE "${INFINI_RT_EXTRA_INCLUDE_DIR}")
        endif()
    endforeach()
endif()

if(INFINI_RT_EXTRA_LIBRARY_PATHS)
    string(REPLACE ":" ";" INFINI_RT_EXTRA_LIBRARY_DIRS
           "${INFINI_RT_EXTRA_LIBRARY_PATHS}")
    foreach(INFINI_RT_EXTRA_LIBRARY_DIR IN LISTS INFINI_RT_EXTRA_LIBRARY_DIRS)
        if(EXISTS "${INFINI_RT_EXTRA_LIBRARY_DIR}")
            target_link_options(install_consumer
                PRIVATE "-Wl,-rpath-link,${INFINI_RT_EXTRA_LIBRARY_DIR}")
        endif()
    endforeach()
endif()
]=])

execute_process(
    COMMAND "${CMAKE_COMMAND}"
            -S "${INFINI_RT_CONSUMER_PROJECT_DIR}"
            -B "${INFINI_RT_CONSUMER_BUILD_DIR}"
            "-DCMAKE_CXX_COMPILER=${INFINI_RT_CXX_COMPILER}"
            "-DCMAKE_PREFIX_PATH=${INFINI_RT_INSTALL_PREFIX}"
            "-DINFINI_RT_INCLUDE_DIR=${INFINI_RT_INCLUDE_DIR}"
            "-DINFINI_RT_LIBRARY_DIR=${INFINI_RT_LIBRARY_DIR}"
            "-DINFINI_RT_CONSUMER_SOURCE=${INFINI_RT_CONSUMER_SOURCE}"
            "-DINFINI_RT_CONSUMER_OUTPUT_DIR=${INFINI_RT_CONSUMER_OUTPUT_DIR}"
            "-DINFINI_RT_CONSUMER_OUTPUT_NAME=${INFINI_RT_CONSUMER_OUTPUT_NAME}"
            "-DINFINI_RT_EXTRA_INCLUDE_PATHS=${INFINI_RT_EXTRA_INCLUDE_PATHS}"
            "-DINFINI_RT_EXTRA_LIBRARY_PATHS=${INFINI_RT_EXTRA_LIBRARY_PATHS}"
            "-DINFINI_RT_CONSUMER_BACKEND=${INFINI_RT_CONSUMER_BACKEND}"
    RESULT_VARIABLE INFINI_RT_COMPILE_RESULT
    OUTPUT_VARIABLE INFINI_RT_COMPILE_OUTPUT
    ERROR_VARIABLE INFINI_RT_COMPILE_ERROR)

if(NOT INFINI_RT_COMPILE_RESULT EQUAL 0)
    message(FATAL_ERROR
            "Configuring the install consumer failed.\n"
            "${INFINI_RT_COMPILE_OUTPUT}\n${INFINI_RT_COMPILE_ERROR}")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" --build "${INFINI_RT_CONSUMER_BUILD_DIR}"
    RESULT_VARIABLE INFINI_RT_COMPILE_RESULT
    OUTPUT_VARIABLE INFINI_RT_COMPILE_OUTPUT
    ERROR_VARIABLE INFINI_RT_COMPILE_ERROR)

if(NOT INFINI_RT_COMPILE_RESULT EQUAL 0)
    message(FATAL_ERROR
            "Compiling the install consumer failed.\n"
            "${INFINI_RT_COMPILE_OUTPUT}\n${INFINI_RT_COMPILE_ERROR}")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env
            "LD_LIBRARY_PATH=${INFINI_RT_LD_LIBRARY_PATH}:$ENV{LD_LIBRARY_PATH}"
            "${INFINI_RT_CONSUMER_BINARY}"
    RESULT_VARIABLE INFINI_RT_RUN_RESULT
    OUTPUT_VARIABLE INFINI_RT_RUN_OUTPUT
    ERROR_VARIABLE INFINI_RT_RUN_ERROR)

if(NOT INFINI_RT_RUN_RESULT EQUAL 0)
    message(FATAL_ERROR
            "Running the install consumer failed.\n"
            "${INFINI_RT_RUN_OUTPUT}\n${INFINI_RT_RUN_ERROR}")
endif()
