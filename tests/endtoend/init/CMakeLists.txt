set(INIT_TESTS
    initarray
    init_expr_call
    init_expr_unary
    initlist_param
    initlist
    initlisttrunc
    initsequence
    init
    inits
)

add_endtoend_tests(EndToEnd.init ${INIT_TESTS})
