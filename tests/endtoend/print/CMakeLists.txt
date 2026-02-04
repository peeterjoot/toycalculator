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

add_endtoend_tests(EndToEnd.print ${PRINT_TESTS})
