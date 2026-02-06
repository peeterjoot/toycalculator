set(EXPRESSIONS_TESTS
    expressions1
    expressions2
    expressions3
    expressions4a
    expressions4b
    expressions4
    expressions5
    expressions6
    expressions_assignment
    expressions_in_call
    expressions_in_for
    expressions_in_if
    bin
    0minus
    chained_comparison_parens
    addi
    converti
    conversions
    neg
    test
)

add_endtoend_tests(EndToEnd.expressions ${EXPRESSIONS_TESTS})
