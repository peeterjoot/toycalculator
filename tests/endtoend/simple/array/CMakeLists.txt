set(ARRAY_TESTS
    array_assign_and_use.silly
    array_dynamic_index.silly
    array_elem_as_arg.silly
    array_in_comparison.silly
    array_index_var.silly
    array_in_expr_min.silly
    array_in_expr.silly
    array_lvalue_complex.silly
    array_mixed_type.silly
    array.silly
    assign_to_array_complex.silly
    intarray.silly
    unary_on_array_elem.silly
    a2.silly
)

# Build the tests
add_endtoend_compile_tests(${ARRAY_TESTS})

# Run and validate the tests
add_endtoend_run_tests("EndToEnd.simple.array" ${ARRAY_TESTS})
