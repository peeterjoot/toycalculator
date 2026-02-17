set(IFELIFELSE_TESTS
    elif2.silly
    elif3.silly
    elif4.silly
    elif5.silly
    elif6.silly
    elif.silly
    ifdcl.silly
    if.silly
    minimal_eliftest.silly
    nested_if_basic.silly
    nested_if_deep.silly
    nested_if_elif_else.silly
)

# Build the tests
add_endtoend_compile_tests(${IFELIFELSE_TESTS})

# Run and validate the tests
add_endtoend_run_tests("EndToEnd.simple.ifelifelse"
    EXPECTED_EXIT_CODE 0
    TESTS ${IFELIFELSE_TESTS}
)
