set(IFELIFELSE_TESTS
    elif2
    elif3
    elif4
    elif5
    elif6
    elif
    ifdcl
    if
    minimal_eliftest
    nested_if_basic
    nested_if_deep
    nested_if_elif_else
)

add_endtoend_tests(EndToEnd.ifelifelse ${IFELIFELSE_TESTS})
