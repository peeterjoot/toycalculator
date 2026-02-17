# sir tests -- not part of regression:
#forbreak
#forcontinue
#forcontinue2

# This one is a dwarfdump test
set(DWARFDUMP_TESTS
    for_simplest
)

add_endtoend_tests("EndToEnd.dwarfdump" ${DWARFDUMP_TESTS})
