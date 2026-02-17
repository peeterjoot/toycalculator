set(BOOL_TESTS
    bool_arith_mixed.silly
    boolcall0.silly
    boolcall1.silly
    boolcall2.silly
    bool_in_arith.silly
    boolliteral.silly
    bool_not_on_comparison.silly
    boolop.silly
    init_expr_bool.silly
)

# Build the tests
add_endtoend_compile_tests(${BOOL_TESTS})

# Run and validate the tests
add_endtoend_run_tests("EndToEnd.simple.bool" ${BOOL_TESTS})
