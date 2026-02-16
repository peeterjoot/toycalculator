set(INIT_TESTS
    init_expr_call
    init_expr_unary
    initlist_param
    initlisttrunc
    initsequence
    init
    inits
)

add_endtoend_tests(EndToEnd.init ${INIT_TESTS})
