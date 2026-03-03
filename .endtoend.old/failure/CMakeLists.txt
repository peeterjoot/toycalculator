
#error_nonconst_init -- want this to be a compile error, but it isn't yet.
#error_zero_step_for -- infinite loops.  would be better to have runtime check for that and abort.
#error_invalid_unary -- regression. was triggering on y undeclared, not on the parse error -- which doesn't actually drive a compile error!
#error_intarray_bad_constaccess
#error_array_oob_const -- dup?
#error_decl_in_if -- probably want this to fail (like scopebug)
#error_decl_in_elif_else -- probably also want this to fail too.
#error_invalid_unary => q(line 3:4 token recognition error at:),


set(FAILURE_TESTS
)

add_endtoend_failure_tests("EndToEnd.failure" ${FAILURE_TESTS})
