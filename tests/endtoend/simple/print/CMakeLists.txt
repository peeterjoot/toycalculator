set(PRINT_TESTS
    printboollit
    printcont
    printdi_nested
    printdi
    printexpr
    printlit
    print_multiple
    printsome
)

add_endtoend_tests(EndToEnd.print ${PRINT_TESTS})
