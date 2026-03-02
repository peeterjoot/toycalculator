# i.e.: ninja check-silly-syntaxtests
add_lit_testsuite(check-silly-syntaxtests "Running silly syntax parse tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS silly
    ARGS -v
)

add_test(
    NAME silly-syntaxtests
    COMMAND ${LLVM_LIT_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}
)
