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

add_endtoend_tests(EndToEnd.operators ${OPERATORS_TESTS})
