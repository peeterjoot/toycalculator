set(DRIVER_TESTS
    twosilly
    onesilly
)

#-------------------------------------------------
# test two source compile and link
add_custom_command(
    OUTPUT
        "${CMAKE_CURRENT_BINARY_DIR}/twosilly"
    COMMAND
        $<TARGET_FILE:silly>
        --output-directory "${CMAKE_CURRENT_BINARY_DIR}"
        -g
        -o twosilly
        ${CMAKE_CURRENT_SOURCE_DIR}/caller.silly
        ${CMAKE_CURRENT_SOURCE_DIR}/callee.silly
    DEPENDS silly ${CMAKE_CURRENT_SOURCE_DIR}/caller.silly ${CMAKE_CURRENT_SOURCE_DIR}/callee.silly
    COMMENT "Compiling twosilly"
    VERBATIM
)

# Create a target for this test so it gets built
add_custom_target(compile_twosilly
    DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/twosilly"
)

# Add to the "all" target so it builds by default
add_dependencies(endtoend_tests compile_twosilly)
#-------------------------------------------------
# test -c/-o, with separate link
add_custom_command(
    OUTPUT
        "${CMAKE_CURRENT_BINARY_DIR}/onesilly"
        "${CMAKE_CURRENT_BINARY_DIR}/one.o"
    COMMAND
        bash -c "$<TARGET_FILE:silly> --output-directory ${CMAKE_CURRENT_BINARY_DIR} -g -c ${CMAKE_CURRENT_SOURCE_DIR}/one.silly ; $<TARGET_FILE:silly> ${CMAKE_CURRENT_BINARY_DIR}/one.o -g -o ${CMAKE_CURRENT_BINARY_DIR}/onesilly"
    DEPENDS silly ${CMAKE_CURRENT_SOURCE_DIR}/one.silly
    COMMENT "Compiling onesilly"
    VERBATIM
)

# Create a target for this test so it gets built
add_custom_target(compile_onesilly
    DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/onesilly"
)

# Add to the "all" target so it builds by default
add_dependencies(endtoend_tests compile_onesilly)
#-------------------------------------------------

add_endtoend_run_tests("EndToEnd.driver"
    EXPECTED_EXIT_CODE 0
    TESTS ${DRIVER_TESTS}
)
