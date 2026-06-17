add_library(infinirt SHARED)

include(GNUInstallDirs)

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

    set_target_properties(infinirt PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
    )
endif()

if(WITH_METAX)
    target_compile_definitions(infinirt PRIVATE WITH_METAX=1)

    target_include_directories(infinirt PUBLIC "${MACA_PATH}/include")
    target_link_libraries(infinirt PUBLIC ${MACA_RUNTIME_LIB})
endif()

if(WITH_MOORE)
    target_compile_definitions(infinirt PRIVATE WITH_MOORE=1)

    target_include_directories(infinirt PUBLIC "${MUSA_ROOT}/include")
    target_link_libraries(infinirt PUBLIC ${MUSA_LIB} ${MUSART_LIB})
endif()

if(WITH_CAMBRICON)
    target_compile_definitions(infinirt PRIVATE WITH_CAMBRICON=1)

    target_include_directories(infinirt PUBLIC "${NEUWARE_HOME}/include")
    target_link_libraries(infinirt PUBLIC ${CAMBRICON_RUNTIME_LIB})
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

    target_include_directories(infinirt PUBLIC
        "${ASCEND_HOME}/include"
        "${ASCEND_HOME}/include/aclnn"
        "${ASCEND_HOME}/include/aclnnop")
    target_link_libraries(infinirt PUBLIC
        "${ASCEND_HOME}/lib64/libascendcl.so"
        "${ASCEND_HAL_LIB}")
endif()

target_include_directories(infinirt
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/generated/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

install(TARGETS infinirt
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/generated/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

file(GLOB INFINI_RT_PUBLIC_CORE_HEADERS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h")

install(FILES ${INFINI_RT_PUBLIC_CORE_HEADERS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/common/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/common
    FILES_MATCHING PATTERN "*.h"
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/native/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/native
    FILES_MATCHING PATTERN "*.h"
)
