set(ARRAY_TESTS
    array_assign_and_use
    array_dynamic_index
    array_elem_as_arg
    array_in_comparison
    array_index_var
    array_in_expr_min
    array_in_expr
    array_lvalue_complex
    array_mixed_type
    array
    assign_to_array_complex
    intarray
    unary_on_array_elem
    a2
)

add_endtoend_tests(EndToEnd.array ${ARRAY_TESTS})
