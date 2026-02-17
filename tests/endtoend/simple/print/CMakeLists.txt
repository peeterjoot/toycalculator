set(PRINT_TESTS
    printboollit.silly
    printcont.silly
    printdi_nested.silly
    printdi.silly
    printexpr.silly
    printlit.silly
    print_multiple.silly
    printsome.silly
)

# Build the tests
add_endtoend_compile_tests(${PRINT_TESTS})

# Run and validate the tests
add_endtoend_run_tests("EndToEnd.simple.print"
    EXPECTED_EXIT_CODE 0
    TESTS ${PRINT_TESTS}
)
