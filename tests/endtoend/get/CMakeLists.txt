set(GET_TESTS
    get.silly
    getboolfail.silly
    getbool.silly
)

add_endtoend_compile_tests(${GET_TESTS})

# Test with input file
add_endtoend_run_tests("EndToEnd.get"
    INPUT_FILE input/get.in
    TESTS
        get.silly
)

add_endtoend_run_tests("EndToEnd.get"
    INPUT_FILE input/getbool.in
    TESTS
        getbool.silly
)

# Test with input and expected abort
add_endtoend_run_tests("EndToEnd.complex.bool"
    INPUT_FILE input/getboolfail.in
    EXPECTED_ABORT
    TESTS
        getboolfail.silly
)
