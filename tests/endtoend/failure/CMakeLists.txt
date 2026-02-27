
#error_nonconst_init -- want this to be a compile error, but it isn't yet.
#error_zero_step_for -- infinite loops.  would be better to have runtime check for that and abort.
#error_invalid_unary -- regression. was triggering on y undeclared, not on the parse error -- which doesn't actually drive a compile error!
#error_intarray_bad_constaccess
#error_array_oob_const -- dup?
#error_decl_in_if -- probably want this to fail (like scopebug)
#error_decl_in_elif_else -- probably also want this to fail too.
#'error_invalid_unary' => q(line 3:4 token recognition error at:),

set(COMPILER_FLAGS_error_arrayreturn --no-verbose-parse-error)

set(FAILURE_TESTS
    bad_call_num_params.silly
    error_arrayreturn.silly
    error_array_too_many_init.silly
    error_bad_return_elif.silly
    error_bad_return_else.silly
    error_bad_return_for.silly
    error_bad_return_if.silly
    error_chained_comparison2.silly
    error_chained_equality.silly
    error_expressions4c.silly
    error_expressions4d.silly
    error_expressions4e.silly
    error_for1.silly
    error_for2.silly
    error_function_redeclare.silly
    error_function_redefine.silly
    error_get_string.silly
    error_induction_var_in_step.silly
    error_init_list1.silly
    error_init_list2.silly
    error_invalid_binary.silly
    error_keyword_declare2.silly
    error_keyword_declare.silly
    error_multiple_in_func.silly
    error_nested_if.silly
    error_nested_ivar_conflict.silly
    error_nested.silly
    error_notfloat.silly
    error_redeclare.silly
    error_return_expr_no_return_type.silly
    error_scopebug.silly
    error_shadow_induction.silly
    error_stringreturn.silly
    error_triple_nested_for_with_shadowing.silly
    error_undeclare.silly
    function_without_return.silly
)

add_endtoend_failure_tests("EndToEnd.failure" ${FAILURE_TESTS})
