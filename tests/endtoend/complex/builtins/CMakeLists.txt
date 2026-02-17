set(COMPLEX_BUILTINS_TESTS
    fatal
    get
)

add_endtoend_tests("EndToEnd.complex.builtins" ${COMPLEX_BUILTINS_TESTS})
