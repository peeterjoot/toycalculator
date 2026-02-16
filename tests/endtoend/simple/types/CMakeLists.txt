set(TYPES_TESTS
    mixed_type_assignment
    types
    const_overflow
    dcl_assign
    dcl
    loadstore
)

add_endtoend_tests(EndToEnd.types ${TYPES_TESTS})
