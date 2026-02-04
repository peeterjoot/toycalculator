set(IFELIFELSE_TESTS
    elif2.silly
    elif3.silly
    elif4.silly
    elif5.silly
    elif6.silly
    elif.silly
    ifdcl.silly
    if.silly
    minimal_eliftest.silly
    nested_if_basic.silly
    nested_if_deep.silly
    nested_if_elif_else.silly
)

add_endtoend_tests(EndToEnd.ifelifelse ${IFELIFELSE_TESTS})
