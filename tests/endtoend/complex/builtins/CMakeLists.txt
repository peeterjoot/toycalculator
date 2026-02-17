# RC:
# 'fatal' => 134, # abort
#
# INPUT:
# 'get' => '6 7 1 2 3 4 5',
set(COMPLEX_BUILTINS_TESTS
    fatal
    get
)

add_endtoend_tests("EndToEnd.complex.builtins" ${COMPLEX_BUILTINS_TESTS})
