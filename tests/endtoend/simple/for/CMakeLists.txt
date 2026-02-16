set(FOR_TESTS
    for
    for_complex_body
    for_simplest
    forward_ref_init
    induction_not_used
    negative_step_for
    nested_for
    zero_iter_for
)

add_endtoend_tests(EndToEnd.for ${FOR_TESTS})
