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

execute_process(
    COMMAND "${INFINI_RT_CXX_COMPILER}"
            -std=c++17
            -Werror
            "-I${INFINI_RT_INCLUDE_DIR}"
            "${INFINI_RT_CONSUMER_SOURCE}"
            "-L${INFINI_RT_LIBRARY_DIR}"
            -linfinirt
            "-Wl,-rpath,${INFINI_RT_LIBRARY_DIR}"
            -o "${INFINI_RT_CONSUMER_BINARY}"
    RESULT_VARIABLE INFINI_RT_COMPILE_RESULT
    OUTPUT_VARIABLE INFINI_RT_COMPILE_OUTPUT
    ERROR_VARIABLE INFINI_RT_COMPILE_ERROR)

if(NOT INFINI_RT_COMPILE_RESULT EQUAL 0)
    message(FATAL_ERROR
            "Compiling the install consumer failed.\n"
            "${INFINI_RT_COMPILE_OUTPUT}\n${INFINI_RT_COMPILE_ERROR}")
endif()

execute_process(
    COMMAND "${INFINI_RT_CONSUMER_BINARY}"
    RESULT_VARIABLE INFINI_RT_RUN_RESULT
    OUTPUT_VARIABLE INFINI_RT_RUN_OUTPUT
    ERROR_VARIABLE INFINI_RT_RUN_ERROR)

if(NOT INFINI_RT_RUN_RESULT EQUAL 0)
    message(FATAL_ERROR
            "Running the install consumer failed.\n"
            "${INFINI_RT_RUN_OUTPUT}\n${INFINI_RT_RUN_ERROR}")
endif()
