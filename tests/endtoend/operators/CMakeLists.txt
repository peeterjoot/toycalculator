set(OPERATORS_TESTS
    lessbug2
    lessbug3
    lessbug
    lesseqop
    lessop
    not_on_expr
    not
    neqop
    eqop
    andop
    orop
    xorop
    unary
)

add_endtoend_tests(EndToEnd.operators ${OPERATORS_TESTS})
