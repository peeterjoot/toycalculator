set(FUNCTION_TESTS
    function_add
    function_foo_with_local
    function_intret_intparam
    function_intret_void
    function_plist_nocast
    function_plist
    function_return
    function
    function_two
    function_void_intparm
    function_void_void
    return_complex
    return_expression
    call_in_predicate
    call_with_param_referenced
    nested_calls
    scopebug
    factorial2
    factorial
    minmax
)

add_endtoend_tests(EndToEnd.function ${FUNCTION_TESTS})
