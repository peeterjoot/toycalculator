# Allow user override
set(LLVM_LIT_EXECUTABLE "$ENV{HOME}/build-llvm/bin/llvm-lit"
  CACHE FILEPATH "Path to llvm-lit executable")

if(NOT EXISTS "${LLVM_LIT_EXECUTABLE}")
    message(WARNING "llvm-lit not found at ${LLVM_LIT_EXECUTABLE} — skipping dialect parse tests")
    return()
endif()

message(STATUS "llvm-lit set to ${LLVM_LIT_EXECUTABLE}")

set(LLVM_EXTERNAL_LIT "${LLVM_LIT_EXECUTABLE}" CACHE FILEPATH
    "Path to llvm-lit used by LLVM lit targets" FORCE)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_BINARY_DIR}/lit.site.cfg.py
    @ONLY
)

message(STATUS "lit.site.cfg.py set to ${CMAKE_BINARY_DIR}/lit.site.cfg.py")

# i.e.: ninja check-silly-lit-tests
add_lit_testsuite(check-silly-lit-tests "Running silly lit tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS silly
    ARGS -v
)

add_test(
    NAME silly-lit-tests
    COMMAND ${LLVM_LIT_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}
)
