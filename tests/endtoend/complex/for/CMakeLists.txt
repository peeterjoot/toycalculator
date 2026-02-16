# .sir tests
set(COMPLEX_FOR_TESTS
    forbreak
    forcontinue
    forcontinue2
)

add_endtoend_tests(EndToEnd.complex_for ${COMPLEX_FOR_TESTS})
