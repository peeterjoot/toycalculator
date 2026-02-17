set(EXPRESSIONS_TESTS
    0minus.silly
    addi.silly
    bin.silly
    chained_comparison_parens.silly
    conversions.silly
    converti.silly
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
    mod2.silly
    modint.silly
    mod.silly
    mult.silly
    neg.silly
    twobinary.silly
)

# Build the tests
add_endtoend_compile_tests(${EXPRESSIONS_TESTS})

# Run and validate the tests
add_endtoend_run_tests("EndToEnd.simple.expressions" ${EXPRESSIONS_TESTS})
