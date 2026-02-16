set(COMPLEX_TYPES_TESTS
    int32
    two_declare_no_init
)

add_endtoend_tests(EndToEnd.complex_types ${COMPLEX_TYPES_TESTS})
