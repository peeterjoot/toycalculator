set(COMPILER_FLAGS_fatal    --no-abort-path)

set(FATAL_TESTS
    fatal.silly
)

# Build the tests
add_endtoend_compile_tests(${FATAL_TESTS})

add_endtoend_run_tests("EndToEnd.fatal"
    EXPECTED_ABORT
    TESTS fatal.silly
)
