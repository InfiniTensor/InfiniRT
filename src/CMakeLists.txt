add_library(infinirt SHARED)
add_library(InfiniRT::infinirt ALIAS infinirt)

function(_infinirt_use_backend_runtime include_dirs link_libraries)
    add_library(InfiniRT::BackendRuntime INTERFACE IMPORTED GLOBAL)
    if(include_dirs)
        set_target_properties(InfiniRT::BackendRuntime PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${include_dirs}")
    endif()
    set_target_properties(InfiniRT::BackendRuntime PROPERTIES
        INTERFACE_LINK_LIBRARIES "${link_libraries}")
    target_link_libraries(infinirt PUBLIC InfiniRT::BackendRuntime)
endfunction()

file(GLOB BASE_SRCS CONFIGURE_DEPENDS "*.cc")
target_sources(infinirt PRIVATE
    ${BASE_SRCS}
    ${PROJECT_SOURCE_DIR}/generated/src/runtime_dispatch.cc)

if(WITH_CPU)
    target_compile_definitions(infinirt PUBLIC WITH_CPU=1)

    find_package(OpenMP REQUIRED)
    target_link_libraries(infinirt PRIVATE OpenMP::OpenMP_CXX)
endif()

if(WITH_NVIDIA)
    enable_language(CUDA)

    target_compile_definitions(infinirt PUBLIC WITH_NVIDIA=1)

    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(infinirt PUBLIC CUDA::cudart)

    set_target_properties(infinirt PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
    )
endif()

if(WITH_ILUVATAR)
    enable_language(CUDA)

    target_compile_definitions(infinirt PUBLIC WITH_ILUVATAR=1)

    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(infinirt PUBLIC CUDA::cudart)

    set_target_properties(infinirt PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
    )
endif()

if(WITH_HYGON)
    enable_language(CUDA)

    target_compile_definitions(infinirt PUBLIC WITH_HYGON=1)

    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(infinirt PUBLIC CUDA::cudart)

    # The public Hygon driver header exposes cuModuleLoadData, cuLaunchKernel
    # and the other driver entry points, which live in the CUDA driver
    # library, not in cudart. FindCUDAToolkit does not provide CUDA::cuda for
    # every DTK layout, so locate the library here and carry it through
    # InfiniRT::BackendRuntime: the package config re-finds it on the
    # consumer's machine, and the install-consumer test links it by
    # referencing the driver entry points.
    find_library(HYGON_CUDA_DRIVER_LIBRARY
        NAMES cuda
        HINTS "${HYGON_CUDA_ROOT}/lib64/stubs" "${HYGON_CUDA_ROOT}/lib64")
    if(NOT HYGON_CUDA_DRIVER_LIBRARY)
        message(FATAL_ERROR
                "The Hygon CUDA driver library (libcuda) was not found under "
                "'${HYGON_CUDA_ROOT}'. It provides the driver entry points of "
                "the public Hygon driver header.")
    endif()
    _infinirt_use_backend_runtime("" "${HYGON_CUDA_DRIVER_LIBRARY}")

    set_target_properties(infinirt PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
    )
endif()

if(WITH_THEAD)
    enable_language(CUDA)

    target_compile_definitions(infinirt PUBLIC WITH_THEAD=1)

    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(infinirt PUBLIC CUDA::cudart)

    set_target_properties(infinirt PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
    )
endif()

if(WITH_METAX)
    target_compile_definitions(infinirt PRIVATE WITH_METAX=1)
    _infinirt_use_backend_runtime(
        "${MACA_PATH}/include"
        "${MACA_RUNTIME_LIB}")
endif()

if(WITH_MARS)
    target_compile_definitions(infinirt PRIVATE WITH_MARS=1)
    _infinirt_use_backend_runtime(
        "${HPCC_PATH}/include"
        "${HPCC_RUNTIME_LIB}")
endif()

if(WITH_MOORE)
    target_compile_definitions(infinirt PRIVATE WITH_MOORE=1)
    _infinirt_use_backend_runtime(
        "${MUSA_ROOT}/include"
        "${MUSA_LIB};${MUSART_LIB}")
endif()

if(WITH_CAMBRICON)
    target_compile_definitions(infinirt PRIVATE WITH_CAMBRICON=1)
    _infinirt_use_backend_runtime(
        "${NEUWARE_HOME}/include"
        "${CAMBRICON_RUNTIME_LIB}")
endif()

if(WITH_KUNLUN)
    find_package(Threads REQUIRED)
    target_link_libraries(infinirt PUBLIC Threads::Threads)
    target_compile_definitions(infinirt PRIVATE WITH_KUNLUN=1)
    _infinirt_use_backend_runtime(
        "${KUNLUN_INCLUDE_DIR}" "${KUNLUN_RUNTIME_LIB}")
endif()

if(WITH_ASCEND)
    # ASCEND_HOME is set by the top-level CMakeLists.txt.
    target_compile_definitions(infinirt PUBLIC WITH_ASCEND=1)

    # Resolve the driver lib dir two levels above the toolkit root.
    get_filename_component(ASCEND_ROOT "${ASCEND_HOME}/../.." ABSOLUTE)

    # Prefer the real driver HAL; fall back to the toolkit stub for build-only
    # environments (e.g., Docker CI images without hardware drivers installed).
    # CANN <= 8.0: stub at runtime/lib64/stub/; CANN >= 8.5: devlib/<arch>-linux/devlib/.
    set(ASCEND_HAL_REAL   "${ASCEND_ROOT}/driver/lib64/driver/libascend_hal.so")
    set(ASCEND_HAL_STUB   "${ASCEND_HOME}/runtime/lib64/stub/libascend_hal.so")
    set(ASCEND_HAL_DEVLIB "${ASCEND_HOME}/${CMAKE_SYSTEM_PROCESSOR}-linux/devlib/libascend_hal.so")
    if(EXISTS "${ASCEND_HAL_REAL}")
        set(ASCEND_HAL_LIB "${ASCEND_HAL_REAL}")
    elseif(EXISTS "${ASCEND_HAL_STUB}")
        set(ASCEND_HAL_LIB "${ASCEND_HAL_STUB}")
        message(STATUS "ascend_hal: driver not found, using stub for linking")
    elseif(EXISTS "${ASCEND_HAL_DEVLIB}")
        set(ASCEND_HAL_LIB "${ASCEND_HAL_DEVLIB}")
        message(STATUS "ascend_hal: driver not found, using devlib for linking")
    else()
        message(FATAL_ERROR "libascend_hal.so not found (tried ${ASCEND_HAL_REAL}, ${ASCEND_HAL_STUB}, and ${ASCEND_HAL_DEVLIB})")
    endif()

    set(INFINI_RT_ASCEND_INCLUDE_DIRS "${ASCEND_HOME}/include")
    foreach(_include_subdir aclnn aclnnop)
        if(IS_DIRECTORY "${ASCEND_HOME}/include/${_include_subdir}")
            list(APPEND INFINI_RT_ASCEND_INCLUDE_DIRS
                "${ASCEND_HOME}/include/${_include_subdir}")
        endif()
    endforeach()
    _infinirt_use_backend_runtime(
        "${INFINI_RT_ASCEND_INCLUDE_DIRS}"
        "${ASCEND_HOME}/lib64/libascendcl.so;${ASCEND_HAL_LIB};${CMAKE_DL_LIBS}")
endif()

target_include_directories(infinirt
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/generated/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_features(infinirt PUBLIC cxx_std_17)

install(TARGETS infinirt
    EXPORT InfiniRTTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(EXPORT InfiniRTTargets
    FILE InfiniRTTargets.cmake
    NAMESPACE InfiniRT::
    DESTINATION ${INFINI_RT_INSTALL_CMAKEDIR}
)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/generated/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
