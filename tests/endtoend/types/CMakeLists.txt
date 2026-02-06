set(TYPES_TESTS
    mixed_type_assignment
    types
    int32
    const_overflow
    dcl_assign
    dcl
    two_declare_no_init
    loadstore
)

add_endtoend_tests(EndToEnd.types ${TYPES_TESTS})
