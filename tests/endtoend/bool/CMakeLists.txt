set(BOOL_TESTS
    bool_arith_mixed.silly
    boolcall0.silly
    boolcall1.silly
    boolcall2.silly
    bool_in_arith.silly
    boolliteral.silly
    bool_not_on_comparison.silly
    boolop.silly
    boolr.silly
    bool.silly
    getboolfail.silly
    getbool.silly
    init_expr_bool.silly
)

add_endtoend_tests(EndToEnd.bool ${BOOL_TESTS})
