# 'initlist' => '--init-fill 255',
# 'initarray' => '--init-fill 255',
set(COMPLEX_INIT_TESTS
    initarray
    initlist
)

add_endtoend_tests("EndToEnd.complex.init" ${COMPLEX_INIT_TESTS})
