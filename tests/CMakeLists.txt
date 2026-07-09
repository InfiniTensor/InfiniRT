if(INFINI_RT_BUILD_TESTING)

function(add_infini_rt_test target)
    add_executable(${target} ${ARGN})
    target_link_libraries(${target} PRIVATE infinirt)
    add_test(NAME ${target} COMMAND ${target})
endfunction()

function(add_infini_rt_backend_runtime_test backend device_type runtime_header
         expect_async_memcpy_success supports_host_memory supports_async_memory
         supports_mem_get_info supports_memset_async supports_stream_wait_event
         supports_event supports_event_elapsed_time)
    string(TOLOWER "${backend}" backend_lower)
    set(target "test_${backend_lower}_runtime")
    add_infini_rt_test(${target} test_native_runtime.cc)
    target_compile_definitions(${target}
        PRIVATE
            "INFINI_RT_TEST_BACKEND_NAME=\"${backend}\""
            "INFINI_RT_TEST_DEVICE_TYPE=${device_type}"
            "INFINI_RT_TEST_RUNTIME_HEADER=\"${runtime_header}\""
            "INFINI_RT_TEST_EXPECT_ASYNC_MEMCPY_SUCCESS=${expect_async_memcpy_success}"
            "INFINI_RT_TEST_SUPPORTS_HOST_MEMORY=${supports_host_memory}"
            "INFINI_RT_TEST_SUPPORTS_ASYNC_MEMORY=${supports_async_memory}"
            "INFINI_RT_TEST_SUPPORTS_MEM_GET_INFO=${supports_mem_get_info}"
            "INFINI_RT_TEST_SUPPORTS_MEMSET_ASYNC=${supports_memset_async}"
            "INFINI_RT_TEST_SUPPORTS_STREAM_WAIT_EVENT=${supports_stream_wait_event}"
            "INFINI_RT_TEST_SUPPORTS_EVENT=${supports_event}"
            "INFINI_RT_TEST_SUPPORTS_EVENT_ELAPSED_TIME=${supports_event_elapsed_time}")
endfunction()

add_infini_rt_test(test_smoke test_smoke.cc)
add_infini_rt_test(test_core test_core.cc)

set(INFINI_RT_TEST_HAS_RUNTIME_BACKEND OFF)

if(WITH_CPU)
    set(INFINI_RT_TEST_HAS_RUNTIME_BACKEND ON)
    add_infini_rt_backend_runtime_test(
        CPU infini::rt::Device::Type::kCpu infini/rt/cpu/runtime_.h
        0 1 0 1 0 1 1 1)
endif()

if(WITH_NVIDIA)
    set(INFINI_RT_TEST_HAS_RUNTIME_BACKEND ON)
    add_infini_rt_backend_runtime_test(
        NVIDIA infini::rt::Device::Type::kNvidia
        infini/rt/nvidia/runtime_.h
        1 1 1 1 1 1 1 1)
endif()

if(WITH_ILUVATAR)
    set(INFINI_RT_TEST_HAS_RUNTIME_BACKEND ON)
    add_infini_rt_backend_runtime_test(
        ILUVATAR infini::rt::Device::Type::kIluvatar
        infini/rt/iluvatar/runtime_.h
        1 1 1 1 1 1 1 1)
endif()

if(WITH_HYGON)
    set(INFINI_RT_TEST_HAS_RUNTIME_BACKEND ON)
    add_infini_rt_backend_runtime_test(
        HYGON infini::rt::Device::Type::kHygon
        infini/rt/hygon/runtime_.h
        1 1 1 1 1 1 1 1)
endif()

if(WITH_METAX)
    set(INFINI_RT_TEST_HAS_RUNTIME_BACKEND ON)
    add_infini_rt_backend_runtime_test(
        METAX infini::rt::Device::Type::kMetax
        infini/rt/metax/runtime_.h
        1 1 1 1 1 1 1 1)
endif()

if(WITH_MOORE)
    set(INFINI_RT_TEST_HAS_RUNTIME_BACKEND ON)
    add_infini_rt_backend_runtime_test(
        MOORE infini::rt::Device::Type::kMoore
        infini/rt/moore/runtime_.h
        1 1 0 1 1 1 1 1)
endif()

if(WITH_CAMBRICON)
    set(INFINI_RT_TEST_HAS_RUNTIME_BACKEND ON)
    add_infini_rt_backend_runtime_test(
        CAMBRICON infini::rt::Device::Type::kCambricon
        infini/rt/cambricon/runtime_.h
        1 0 0 0 0 0 0 0)
endif()

if(WITH_ASCEND)
    set(INFINI_RT_TEST_HAS_RUNTIME_BACKEND ON)
    add_infini_rt_backend_runtime_test(
        ASCEND infini::rt::Device::Type::kAscend
        infini/rt/ascend/runtime_.h
        1 0 0 0 0 0 0 0)
endif()

if(INFINI_RT_TEST_HAS_RUNTIME_BACKEND)
    add_infini_rt_test(test_runtime_dispatch test_runtime_dispatch.cc)

    if(WITH_CPU)
        target_compile_definitions(test_runtime_dispatch
                                   PRIVATE INFINI_RT_TEST_WITH_CPU=1)
    endif()
    if(WITH_NVIDIA)
        target_compile_definitions(test_runtime_dispatch
                                   PRIVATE INFINI_RT_TEST_WITH_NVIDIA=1)
    endif()
    if(WITH_ILUVATAR)
        target_compile_definitions(test_runtime_dispatch
                                   PRIVATE INFINI_RT_TEST_WITH_ILUVATAR=1)
    endif()
    if(WITH_HYGON)
        target_compile_definitions(test_runtime_dispatch
                                   PRIVATE INFINI_RT_TEST_WITH_HYGON=1)
    endif()
    if(WITH_METAX)
        target_compile_definitions(test_runtime_dispatch
                                   PRIVATE INFINI_RT_TEST_WITH_METAX=1)
    endif()
    if(WITH_MOORE)
        target_compile_definitions(test_runtime_dispatch
                                   PRIVATE INFINI_RT_TEST_WITH_MOORE=1)
    endif()
    if(WITH_CAMBRICON)
        target_compile_definitions(test_runtime_dispatch
                                   PRIVATE INFINI_RT_TEST_WITH_CAMBRICON=1)
    endif()
    if(WITH_ASCEND)
        target_compile_definitions(test_runtime_dispatch
                                   PRIVATE INFINI_RT_TEST_WITH_ASCEND=1)
    endif()
endif()

set(INFINI_RT_TEST_INSTALL_PREFIX
    "${CMAKE_CURRENT_BINARY_DIR}/install_consumer_prefix")
set(INFINI_RT_TEST_CONSUMER_BINARY
    "${CMAKE_CURRENT_BINARY_DIR}/install_consumer_smoke")
set(INFINI_RT_TEST_EXTRA_LIBRARY_DIRS "")
set(INFINI_RT_TEST_EXTRA_INCLUDE_DIRS "")
set(INFINI_RT_TEST_CONSUMER_BACKEND NONE)

if(WITH_NVIDIA OR WITH_ILUVATAR OR WITH_HYGON)
    if(CUDAToolkit_INCLUDE_DIRS)
        list(APPEND INFINI_RT_TEST_EXTRA_INCLUDE_DIRS
             ${CUDAToolkit_INCLUDE_DIRS})
    elseif(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)
        list(APPEND INFINI_RT_TEST_EXTRA_INCLUDE_DIRS
             ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    endif()

    if(CUDAToolkit_LIBRARY_DIR)
        list(APPEND INFINI_RT_TEST_EXTRA_LIBRARY_DIRS
             "${CUDAToolkit_LIBRARY_DIR}")
    endif()
    if(CUDAToolkit_TARGET_DIR)
        list(APPEND INFINI_RT_TEST_EXTRA_LIBRARY_DIRS
             "${CUDAToolkit_TARGET_DIR}/lib64")
    endif()
    if(CUDAToolkit_ROOT_DIR)
        list(APPEND INFINI_RT_TEST_EXTRA_LIBRARY_DIRS
             "${CUDAToolkit_ROOT_DIR}/lib64")
    endif()
endif()

if(WITH_NVIDIA)
    set(INFINI_RT_TEST_CONSUMER_BACKEND NVIDIA)
elseif(WITH_ILUVATAR)
    set(INFINI_RT_TEST_CONSUMER_BACKEND ILUVATAR)
elseif(WITH_HYGON)
    set(INFINI_RT_TEST_CONSUMER_BACKEND HYGON)
    if(HYGON_CUDA_ROOT)
        list(APPEND INFINI_RT_TEST_EXTRA_INCLUDE_DIRS
             "${HYGON_CUDA_ROOT}/include")
        list(APPEND INFINI_RT_TEST_EXTRA_LIBRARY_DIRS
             "${HYGON_CUDA_ROOT}/lib64")
    endif()
elseif(WITH_METAX)
    set(INFINI_RT_TEST_CONSUMER_BACKEND METAX)
    list(APPEND INFINI_RT_TEST_EXTRA_INCLUDE_DIRS
         "${MACA_PATH}/include")
    list(APPEND INFINI_RT_TEST_EXTRA_LIBRARY_DIRS
         "${MACA_PATH}/lib")
elseif(WITH_MOORE)
    set(INFINI_RT_TEST_CONSUMER_BACKEND MOORE)
    list(APPEND INFINI_RT_TEST_EXTRA_INCLUDE_DIRS
         "${MUSA_ROOT}/include")
    list(APPEND INFINI_RT_TEST_EXTRA_LIBRARY_DIRS
         "${MUSA_ROOT}/lib")
elseif(WITH_CAMBRICON)
    set(INFINI_RT_TEST_CONSUMER_BACKEND CAMBRICON)
    list(APPEND INFINI_RT_TEST_EXTRA_INCLUDE_DIRS
         "${NEUWARE_HOME}/include")
    list(APPEND INFINI_RT_TEST_EXTRA_LIBRARY_DIRS
         "${NEUWARE_HOME}/lib"
         "${NEUWARE_HOME}/lib64")
elseif(WITH_ASCEND)
    # The install-consumer smoke checks Ascend public headers and linkage.
    # Runtime behavior is covered by test_ascend_runtime and dispatch tests.
    list(APPEND INFINI_RT_TEST_EXTRA_INCLUDE_DIRS
         "${ASCEND_HOME}/include"
         "${ASCEND_HOME}/include/aclnn"
         "${ASCEND_HOME}/include/aclnnop")
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

endif()

if(INFINI_RT_BUILD_PERFORMANCE_TESTING)
    add_subdirectory(performance)
endif()
