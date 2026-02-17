# - two_declare_no_init -- uses a perl function to check for declaration order in the generated mlir.
set(MLIRPARSE_TESTS
    two_declare_no_init
)

add_endtoend_tests("EndToEnd.mlirparsetest" ${MLIRPARSE_TESTS})
