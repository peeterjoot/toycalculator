set(FUNCTION_TESTS
    call_in_predicate.silly
    call_with_param_referenced.silly
    factorial2.silly
    factorial.silly
    function_add.silly
    function_foo_with_local.silly
    function_intret_intparam.silly
    function_intret_void.silly
    function_plist_nocast.silly
    function_plist.silly
    function_return.silly
    function.silly
    function_two.silly
    function_void_intparm.silly
    function_void_void.silly
    minmax.silly
    nested_calls.silly
    proto1.silly
    proto2.silly
    proto3.silly
    return_complex.silly
    return_expression.silly
)

# Build the tests
add_endtoend_compile_tests(${FUNCTION_TESTS})

# Run and validate the tests
add_endtoend_run_tests("EndToEnd.simple.function"
    EXPECTED_EXIT_CODE 0
    TESTS ${FUNCTION_TESTS}
)
