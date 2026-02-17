set(STRING_TESTS
    initstring.silly
    longstring.silly
    shortstring2.silly
    shortstring3.silly
    shortstring.silly
    stringlit.silly
    s2.silly
)

# Build the tests
add_endtoend_compile_tests(${STRING_TESTS})

# Run and validate the tests
add_endtoend_run_tests("EndToEnd.simple.string"
    EXPECTED_EXIT_CODE 0
    TESTS ${STRING_TESTS}
)
