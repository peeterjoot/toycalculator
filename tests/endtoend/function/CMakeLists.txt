set(FUNCTION_TESTS
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
    return_complex.silly
    return_expression.silly
    call_in_predicate.silly
    call_with_param_referenced.silly
    nested_calls.silly
    scopebug.silly
    factorial2.silly
    factorial.silly
    minmax.silly
)

add_endtoend_tests(EndToEnd.function ${FUNCTION_TESTS})
