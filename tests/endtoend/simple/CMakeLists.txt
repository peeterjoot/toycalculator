set(SIMPLE_TESTS
    simplest.silly
    simpleless.silly
    empty.silly
)

add_endtoend_tests(EndToEnd.simple ${SIMPLE_TESTS})
