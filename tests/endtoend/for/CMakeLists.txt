set(FOR_TESTS
    for_complex_body
    for
    for_simplest
    forward_ref_init
    negative_step_for
    nested_for
    zero_iter_for
    induction_not_used
    forbreak
)

add_endtoend_tests(EndToEnd.for ${FOR_TESTS})
