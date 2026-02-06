set(BUILTINS_TESTS
    fatal
    get
)

add_endtoend_tests(EndToEnd.builtins ${BUILTINS_TESTS})
