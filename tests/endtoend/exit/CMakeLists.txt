#set(EXIT_TESTS
#    bool.silly
#    boolr.silly
#    int32.silly
#    exit3.silly
#    exitx.silly
#    exit42.silly
#    exitarrayelement.silly
#)
#
## Build the tests
#add_endtoend_compile_tests(${EXIT_TESTS})
#
#add_endtoend_run_tests("EndToEnd.exit"
#    EXPECTED_EXIT_CODE 1
#    TESTS
#        bool.silly
#        boolr.silly
#        int32.silly
#)
#
#add_endtoend_run_tests("EndToEnd.exit"
#    EXPECTED_EXIT_CODE 3
#    TESTS
#        exit3.silly
#        exitx.silly
#)
#
#add_endtoend_run_tests("EndToEnd.exit"
#    EXPECTED_EXIT_CODE 42
#    TESTS
#        exit42.silly
#        exitarrayelement.silly
#)

set(EXIT_TESTS
    bool
    boolr
    int32
    exit3
    exitx
    exit42
    exitarrayelement
)

add_endtoend_tests("EndToEnd.exit" ${EXIT_TESTS})
