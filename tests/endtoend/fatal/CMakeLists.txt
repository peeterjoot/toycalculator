#set(FATAL_TESTS
#    fatal.silly
#)
#
## Build the tests
#add_endtoend_compile_tests(${FATAL_TESTS})
#
# Fails because of an absolute path in the stderr -- revisit and see if I can mask that.
#add_endtoend_run_tests("EndToEnd.fatal"
#    EXPECTED_ABORT
#    TESTS
#        fatal.silly
#)

set(FATAL_TESTS
    fatal
)

add_endtoend_tests("EndToEnd.fatal" ${FATAL_TESTS})
