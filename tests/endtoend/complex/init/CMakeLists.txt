set(COMPLEX_INIT_TESTS
    initarray
    initlist
)

add_endtoend_tests("EndToEnd.complex.init" ${COMPLEX_INIT_TESTS})
