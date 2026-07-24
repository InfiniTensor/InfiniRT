if(NOT INFINI_RT_SOURCE_DIR OR NOT INFINI_RT_TEST_BINARY_DIR)
    message(FATAL_ERROR "InfiniRT source and test binary directories are required.")
endif()

set(_hpcc_root "${INFINI_RT_TEST_BINARY_DIR}/hpcc")
set(_maca_root "${INFINI_RT_TEST_BINARY_DIR}/maca")
set(_child_build "${INFINI_RT_TEST_BINARY_DIR}/build")
file(REMOVE_RECURSE "${INFINI_RT_TEST_BINARY_DIR}")
file(MAKE_DIRECTORY "${_hpcc_root}/include/hcr" "${_maca_root}/include/mcr")
file(WRITE "${_hpcc_root}/include/hcr/hc_runtime.h" "")
file(WRITE "${_maca_root}/include/mcr/mc_runtime.h" "")

execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env
            "HPCC_PATH=${_hpcc_root}"
            "MACA_PATH=${_maca_root}"
            "${CMAKE_COMMAND}"
            -S "${INFINI_RT_SOURCE_DIR}"
            -B "${_child_build}"
            -DAUTO_DETECT_DEVICES=ON
            -DINFINI_RT_BUILD_TESTING=OFF
    RESULT_VARIABLE _configure_result
    OUTPUT_VARIABLE _configure_stdout
    ERROR_VARIABLE _configure_stderr)

if(_configure_result EQUAL 0)
    message(FATAL_ERROR "Ambiguous MetaX-family auto-detection unexpectedly succeeded.")
endif()

set(_configure_output "${_configure_stdout}\n${_configure_stderr}")
string(FIND
       "${_configure_output}"
       "Both HPCC_PATH and MACA_PATH identify valid SDKs"
       _expected_error_index)
if(_expected_error_index EQUAL -1)
    message(FATAL_ERROR
            "Ambiguous auto-detection failed for the wrong reason:\n${_configure_output}")
endif()
