set(DEBUG_TESTS
    foo.silly
    f.silly
)

# Build the tests
add_endtoend_compile_tests(${DEBUG_TESTS})

# Run and validate the tests
add_endtoend_run_tests("EndToEnd.simple.debug"
    EXPECTED_EXIT_CODE 0
    TESTS ${DEBUG_TESTS}
)
