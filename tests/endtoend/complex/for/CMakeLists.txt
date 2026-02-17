# sir tests -- not part of regression:
#forbreak
#forcontinue
#forcontinue2

# This one is a dwarfdump test
set(COMPLEX_FOR_TESTS
    for_simplest
)

add_endtoend_tests("EndToEnd.complex.for" ${COMPLEX_FOR_TESTS})
