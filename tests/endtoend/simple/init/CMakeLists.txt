set(INIT_TESTS
    init_expr_call.silly
    init_expr_unary.silly
    initlist_param.silly
    initlisttrunc.silly
    initsequence.silly
    init.silly
    inits.silly
)

# Build the tests
add_endtoend_compile_tests(${INIT_TESTS})

# Run and validate the tests
add_endtoend_run_tests("EndToEnd.simple.init"
    EXPECTED_EXIT_CODE 0
    TESTS ${INIT_TESTS}
)
