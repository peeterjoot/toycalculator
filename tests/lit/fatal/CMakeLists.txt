# i.e.: ninja check-silly-fataltests
add_lit_testsuite(check-silly-fataltests "Running silly fatal tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS silly
    ARGS -v
)

add_test(
    NAME silly-fataltests
    COMMAND ${LLVM_LIT_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}
)
