set(DRIVER_TESTS
    twosource
    onesource
)

#-------------------------------------------------
# two source compile and link
add_endtoend_multisource_compile_tests( twosource caller.silly callee.silly )

# -c/-o, with separate link
add_silly_compile_link_test( onesource one.silly )
#-------------------------------------------------

add_endtoend_run_tests("EndToEnd.driver"
    EXPECTED_EXIT_CODE 0
    TESTS ${DRIVER_TESTS}
)
