set(STRING_TESTS
    initstring
    longstring
    shortstring2
    shortstring3
    shortstring
    stringlit
    s2
)

add_endtoend_tests(EndToEnd.string ${STRING_TESTS})
