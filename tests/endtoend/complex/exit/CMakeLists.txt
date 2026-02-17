# 'exit3' => 3,
# 'exit42' => 42,
# 'exitx' => 3,
set(EXIT_TESTS
    exit3
    exit42
    exitx
)

add_endtoend_tests("EndToEnd.complex.exit" ${EXIT_TESTS})
