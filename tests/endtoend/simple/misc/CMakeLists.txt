#div_zero_int
set(MISC_TESTS
    div_zero_float.silly
)

# Build the tests
add_endtoend_compile_tests(${MISC_TESTS})

# Run and validate the tests
add_endtoend_run_tests("EndToEnd.simple.misc"
    EXPECTED_EXIT_CODE 0
    TESTS ${MISC_TESTS}
)
