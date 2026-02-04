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

add_endtoend_tests(EndToEnd.init ${INIT_TESTS})
