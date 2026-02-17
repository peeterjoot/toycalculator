set(TYPES_TESTS
    mixed_type_assignment.silly
    types.silly
    const_overflow.silly
    dcl_assign.silly
    dcl.silly
    loadstore.silly
)

# Build the tests
add_endtoend_compile_tests(${TYPES_TESTS})

# Run and validate the tests
add_endtoend_run_tests("EndToEnd.simple.types" ${TYPES_TESTS})
