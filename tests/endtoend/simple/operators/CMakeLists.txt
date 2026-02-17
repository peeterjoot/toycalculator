set(OPERATORS_TESTS
    lessbug2.silly
    lessbug3.silly
    lessbug.silly
    lesseqop.silly
    lessop.silly
    not_on_expr.silly
    not.silly
    neqop.silly
    eqop.silly
    andop.silly
    orop.silly
    xorop.silly
    unary.silly
)

# Build the tests
add_endtoend_compile_tests(${OPERATORS_TESTS})

# Run and validate the tests
add_endtoend_run_tests("EndToEnd.simple.operators" ${OPERATORS_TESTS})
