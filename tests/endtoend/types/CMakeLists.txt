set(TYPES_TESTS
    mixed_type_assignment.silly
    types.silly
    int32.silly
    const_overflow.silly
    dcl_assign.silly
    dcl.silly
    two_declare_no_init.silly
    loadstore.silly
)

add_endtoend_tests(EndToEnd.types ${TYPES_TESTS})
