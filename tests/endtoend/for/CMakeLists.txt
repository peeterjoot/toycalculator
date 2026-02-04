set(FOR_TESTS
    for_complex_body.silly
    for.silly
    for_simplest.silly
    forward_ref_init.silly
    negative_step_for.silly
    nested_for.silly
    zero_iter_for.silly
    induction_not_used.silly
)

add_endtoend_tests(EndToEnd.for ${FOR_TESTS})
