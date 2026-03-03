# i.e.: ninja check-silly-gettests
add_lit_testsuite(check-silly-gettests "Running silly get tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS silly
    ARGS -v
)

add_test(
    NAME silly-gettests
    COMMAND ${LLVM_LIT_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}
)
