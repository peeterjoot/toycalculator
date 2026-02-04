set(EXPRESSIONS_TESTS
    expressions1.silly
    expressions2.silly
    expressions3.silly
    expressions4a.silly
    expressions4b.silly
    expressions4.silly
    expressions5.silly
    expressions6.silly
    expressions_assignment.silly
    expressions_in_call.silly
    expressions_in_for.silly
    expressions_in_if.silly
    bin.silly
    0minus.silly
    chained_comparison_parens.silly
    addi.silly
    converti.silly
    conversions.silly
    neg.silly
    test.silly
)

add_endtoend_tests(EndToEnd.expressions ${EXPRESSIONS_TESTS})
