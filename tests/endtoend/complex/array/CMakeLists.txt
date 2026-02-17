
# arrayprod -- requires: --init-fill 255
# exitarrayelement -- RC=42
set(COMPLEX_ARRAY_TESTS
    arrayprod
    exitarrayelement
)

add_endtoend_tests("EndToEnd.complex.array" ${COMPLEX_ARRAY_TESTS})
