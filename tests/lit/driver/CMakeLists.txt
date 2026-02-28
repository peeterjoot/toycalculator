# i.e.: ninja check-silly-drivertests
add_lit_testsuite(check-silly-drivertests "Running silly driver parse tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS silly
    ARGS -v
)

add_test(
    NAME silly-drivertests
    COMMAND ${LLVM_LIT_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}
)
