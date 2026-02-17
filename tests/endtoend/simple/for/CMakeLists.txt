set(FOR_TESTS
    for.silly
    for_complex_body.silly
    forward_ref_init.silly
    negative_step_for.silly
    nested_for.silly
    zero_iter_for.silly
    induction_not_used.silly
)

# Build the tests
add_endtoend_compile_tests(${FOR_TESTS})

# Run and validate the tests
add_endtoend_run_tests("EndToEnd.simple.for"
    EXPECTED_EXIT_CODE 0
    TESTS ${FOR_TESTS}
)
