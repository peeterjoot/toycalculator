set(COMPLEX_BOOL_TESTS
    boolr
    bool
    getboolfail
    getbool
)

add_endtoend_tests("EndToEnd.complex.bool" ${COMPLEX_BOOL_TESTS})
