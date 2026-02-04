set(BUILTINS_TESTS
    fatal.silly
    get.silly
)

add_endtoend_tests(EndToEnd.builtins ${BUILTINS_TESTS})
