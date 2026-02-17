set(COMPLEX_ARRAY_TESTS
    arrayprod
    exitarrayelement
)

add_endtoend_tests("EndToEnd.complex.array" ${COMPLEX_ARRAY_TESTS})
