set(COMPILER_FLAGS_initlist     --init-fill 255)
set(COMPILER_FLAGS_initarray    --init-fill 255)

set(INIT_TESTS
    initarray.silly
    init_expr_call.silly
    init_expr_unary.silly
    initlist_param.silly
    initlist.silly
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
