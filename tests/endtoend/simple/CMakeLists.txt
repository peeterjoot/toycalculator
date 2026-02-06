set(SIMPLE_TESTS
    simplest
    simpleless
    empty
)

add_endtoend_tests(EndToEnd.simple ${SIMPLE_TESTS})
