set(SIMPLE_TESTS
    simplest.silly
    simpleless.silly
    empty.silly
)

# Build the tests
add_endtoend_compile_tests(${SIMPLE_TESTS})

# Run and validate the tests
add_endtoend_run_tests("EndToEnd.simple.simple"
    EXPECTED_EXIT_CODE 0
    TESTS ${SIMPLE_TESTS}
)
