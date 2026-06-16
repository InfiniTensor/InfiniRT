function(add_infini_rt_test target)
    add_executable(${target} ${ARGN})
    target_link_libraries(${target} PRIVATE infinirt)
    add_test(NAME ${target} COMMAND ${target})
endfunction()

add_infini_rt_test(test_smoke test_smoke.cc)
