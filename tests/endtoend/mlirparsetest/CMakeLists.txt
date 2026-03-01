if(FILECHECK_EXECUTABLE)
    set(DECLARE_ORDER_TESTS
        two_declare_no_init.silly
    )

    set(COMPILER_FLAGS_two_declare_no_init --emit-llvm --emit-mlir)

    add_endtoend_compile_tests(${DECLARE_ORDER_TESTS})

    add_endtoend_run_tests("EndToEnd.mlirparsetest"
        EXPECTED_EXIT_CODE 0
        TESTS ${DECLARE_ORDER_TESTS}
    )

    add_test(
        NAME EndToEnd.mlirparsetest.two_declare_no_init.mlirfilecheck
        COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_CURRENT_BINARY_DIR}
            ${FILECHECK_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/two_declare_no_init.check --input-file=${CMAKE_CURRENT_BINARY_DIR}/two_declare_no_init.mlir
    )

    set_tests_properties(EndToEnd.mlirparsetest.two_declare_no_init.mlirfilecheck
        PROPERTIES
            DEPENDS EndToEnd.mlirparsetest.two_declare_no_init
            LABELS "mlir;filecheck"
    )
endif()
