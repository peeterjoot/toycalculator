# i.e.: ninja check-silly-debugtests
add_lit_testsuite(check-silly-debugtests "Running silly debug parse tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS silly
    ARGS -v
)

add_test(
    NAME silly-debugtests
    COMMAND ${LLVM_LIT_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}
)
