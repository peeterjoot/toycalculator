
# arrayprod -- requires: --init-fill 255
set(COMPLEX_ARRAY_TESTS
    arrayprod
)

add_endtoend_tests("EndToEnd.complex.array" ${COMPLEX_ARRAY_TESTS})
