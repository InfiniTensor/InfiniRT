cmake_minimum_required(VERSION 3.18)
project(infinirt_install_consumer LANGUAGES CXX)

if(NOT INFINI_RT_CONSUMER_SOURCE)
    message(FATAL_ERROR "INFINI_RT_CONSUMER_SOURCE is required.")
endif()
if(NOT INFINI_RT_REQUIRED_VERSION)
    message(FATAL_ERROR "INFINI_RT_REQUIRED_VERSION is required.")
endif()

find_package(InfiniRT ${INFINI_RT_REQUIRED_VERSION} EXACT CONFIG REQUIRED)

if(NOT TARGET InfiniRT::infinirt)
    message(FATAL_ERROR "The InfiniRT::infinirt target was not imported.")
endif()
if(NOT InfiniRT_ENABLED_BACKENDS)
    message(FATAL_ERROR "InfiniRT did not report its enabled backends.")
endif()

add_executable(infinirt_install_consumer "${INFINI_RT_CONSUMER_SOURCE}")
target_link_libraries(infinirt_install_consumer PRIVATE InfiniRT::infinirt)

if(INFINI_RT_CONSUMER_BACKEND
   AND NOT INFINI_RT_CONSUMER_BACKEND STREQUAL "NONE")
    target_compile_definitions(
        infinirt_install_consumer
        PRIVATE
            "INFINI_RT_CONSUMER_BACKEND_${INFINI_RT_CONSUMER_BACKEND}=1")
endif()

enable_testing()
add_test(NAME run_infinirt_install_consumer
         COMMAND infinirt_install_consumer)
