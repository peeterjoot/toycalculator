#
# INPUT:
# 'get' => '6 7 1 2 3 4 5',
set(COMPLEX_BUILTINS_TESTS
    get
)

add_endtoend_tests("EndToEnd.complex.builtins" ${COMPLEX_BUILTINS_TESTS})
