cmake_minimum_required(VERSION 3.18)
project(infinirt_consumer_example LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED INFINI_RT_PREFIX)
    message(FATAL_ERROR "Set INFINI_RT_PREFIX to an installed InfiniRT prefix.")
endif()

find_library(INFINI_RT_LIBRARY
    NAMES infinirt
    PATHS "${INFINI_RT_PREFIX}/lib" "${INFINI_RT_PREFIX}/lib64"
    REQUIRED
    NO_DEFAULT_PATH)

add_executable(infinirt_consumer main.cc)
target_include_directories(infinirt_consumer PRIVATE
    "${INFINI_RT_PREFIX}/include")
target_link_libraries(infinirt_consumer PRIVATE "${INFINI_RT_LIBRARY}")

get_filename_component(INFINI_RT_LIBRARY_DIR "${INFINI_RT_LIBRARY}" DIRECTORY)
set_target_properties(infinirt_consumer PROPERTIES
    BUILD_RPATH "${INFINI_RT_LIBRARY_DIR}")

