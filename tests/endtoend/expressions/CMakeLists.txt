set(EXPRESSIONS_TESTS
    0minus
    addi
    bin
    chained_comparison_parens
    conversions
    converti
    expressions1
    expressions2
    expressions3
    expressions4
    expressions4a
    expressions4b
    expressions5
    expressions6
    expressions_assignment
    expressions_in_call
    expressions_in_for
    expressions_in_if
    mult
    neg
    test
)

add_endtoend_tests(EndToEnd.expressions ${EXPRESSIONS_TESTS})
