# - two_declare_no_init -- uses a perl function to check for declaration order in the generated mlir.
# - int32 -- expect RC=1
set(COMPLEX_TYPES_TESTS
    int32
    two_declare_no_init
)

add_endtoend_tests("EndToEnd.complex.types" ${COMPLEX_TYPES_TESTS})
