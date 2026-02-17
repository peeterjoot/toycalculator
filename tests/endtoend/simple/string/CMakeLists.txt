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
add_endtoend_run_tests("EndToEnd.simple.string" ${STRING_TESTS})
