function(add_infini_rt_test target)
    add_executable(${target} ${ARGN})
    target_link_libraries(${target} PRIVATE infinirt)
    add_test(NAME ${target} COMMAND ${target})
endfunction()

add_infini_rt_test(test_smoke test_smoke.cc)
add_infini_rt_test(test_core test_core.cc)
add_infini_rt_test(test_tensor_view test_tensor_view.cc)

if(WITH_CPU OR WITH_NVIDIA)
    add_infini_rt_test(test_runtime_dispatch test_runtime_dispatch.cc)
    if(WITH_CPU)
        target_compile_definitions(test_runtime_dispatch
                                   PRIVATE INFINI_RT_TEST_WITH_CPU=1)
    endif()
    if(WITH_NVIDIA)
        target_compile_definitions(test_runtime_dispatch
                                   PRIVATE INFINI_RT_TEST_WITH_NVIDIA=1)
    endif()
endif()

if(WITH_CPU)
    add_infini_rt_test(test_cpu_runtime test_cpu_runtime.cc)
endif()

if(WITH_NVIDIA)
    add_infini_rt_test(test_nvidia_runtime test_nvidia_runtime.cc)
endif()

set(INFINI_RT_TEST_INSTALL_PREFIX
    "${CMAKE_CURRENT_BINARY_DIR}/install_consumer_prefix")
set(INFINI_RT_TEST_CONSUMER_BINARY
    "${CMAKE_CURRENT_BINARY_DIR}/install_consumer_smoke")
set(INFINI_RT_TEST_EXTRA_LIBRARY_DIRS "")
set(INFINI_RT_TEST_EXTRA_INCLUDE_DIRS "")
set(INFINI_RT_TEST_CONSUMER_BACKEND NONE)

if(WITH_NVIDIA)
    set(INFINI_RT_TEST_CONSUMER_BACKEND NVIDIA)
    if(CUDAToolkit_INCLUDE_DIRS)
        list(APPEND INFINI_RT_TEST_EXTRA_INCLUDE_DIRS
             ${CUDAToolkit_INCLUDE_DIRS})
    elseif(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)
        list(APPEND INFINI_RT_TEST_EXTRA_INCLUDE_DIRS
             ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    endif()
elseif(WITH_CPU)
    set(INFINI_RT_TEST_CONSUMER_BACKEND CPU)
endif()

if(WITH_ASCEND)
    get_filename_component(
        INFINI_RT_ASCEND_DEVLIB
        "${ASCEND_HOME}/${CMAKE_SYSTEM_PROCESSOR}-linux/devlib"
        ABSOLUTE)
    list(APPEND INFINI_RT_TEST_EXTRA_LIBRARY_DIRS
         "${ASCEND_HOME}/lib64"
         "${INFINI_RT_ASCEND_DEVLIB}"
         "${ASCEND_HOME}/runtime/lib64/stub")
endif()

list(JOIN INFINI_RT_TEST_EXTRA_LIBRARY_DIRS ":" INFINI_RT_TEST_EXTRA_LIBRARY_PATHS)
list(JOIN INFINI_RT_TEST_EXTRA_INCLUDE_DIRS ":" INFINI_RT_TEST_EXTRA_INCLUDE_PATHS)

add_test(
    NAME test_install
    COMMAND "${CMAKE_COMMAND}" --install "${CMAKE_BINARY_DIR}"
            --prefix "${INFINI_RT_TEST_INSTALL_PREFIX}")
set_tests_properties(test_install PROPERTIES FIXTURES_SETUP infini_rt_install)

add_test(
    NAME test_install_consumer
    COMMAND "${CMAKE_COMMAND}"
            "-DINFINI_RT_INSTALL_PREFIX=${INFINI_RT_TEST_INSTALL_PREFIX}"
            "-DINFINI_RT_CONSUMER_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/install_consumer_smoke.cc"
            "-DINFINI_RT_CONSUMER_BINARY=${INFINI_RT_TEST_CONSUMER_BINARY}"
            "-DINFINI_RT_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
            "-DINFINI_RT_EXTRA_LIBRARY_PATHS=${INFINI_RT_TEST_EXTRA_LIBRARY_PATHS}"
            "-DINFINI_RT_EXTRA_INCLUDE_PATHS=${INFINI_RT_TEST_EXTRA_INCLUDE_PATHS}"
            "-DINFINI_RT_CONSUMER_BACKEND=${INFINI_RT_TEST_CONSUMER_BACKEND}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/compile_install_consumer.cmake")
set_tests_properties(test_install_consumer PROPERTIES
    FIXTURES_REQUIRED infini_rt_install)
