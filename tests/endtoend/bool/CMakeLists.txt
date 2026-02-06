set(BOOL_TESTS
    bool_arith_mixed
    boolcall0
    boolcall1
    boolcall2
    bool_in_arith
    boolliteral
    bool_not_on_comparison
    boolop
    boolr
    bool
    getboolfail
    getbool
    init_expr_bool
)

add_endtoend_tests(EndToEnd.bool ${BOOL_TESTS})
