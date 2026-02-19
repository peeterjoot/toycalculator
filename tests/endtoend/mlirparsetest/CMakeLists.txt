set(DECLARATION_ORDER_TESTS
    two_declare_no_init.silly
)

add_endtoend_compile_tests(${DECLARATION_ORDER_TESTS})

add_endtoend_run_tests("EndToEnd.mlirparsetest"
    EXPECTED_EXIT_CODE 0
    TESTS ${DECLARATION_ORDER_TESTS}
)

add_test(
    NAME EndToEnd.mlirparsetest.two_declare_no_init.check_decl_order
    COMMAND ${CMAKE_SOURCE_DIR}/bin/check_decl_order
        --what a,b
        --mlirpath ${CMAKE_CURRENT_BINARY_DIR}/two_declare_no_init.mlir
)

set_tests_properties(EndToEnd.mlirparsetest.two_declare_no_init.check_decl_order
    PROPERTIES
        DEPENDS EndToEnd.mlirparsetest.two_declare_no_init
        LABELS "mlirparsetest"
)
