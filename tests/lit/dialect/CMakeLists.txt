# i.e.: ninja check-silly-dialecttests
add_lit_testsuite(check-silly-dialecttests "Running silly dialect parse tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS SillyDialect
    ARGS -v
)

add_test(
    NAME silly-dialecttests
    COMMAND ${LLVM_LIT_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}
)
