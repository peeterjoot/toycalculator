# This script runs a test executable and captures stdout/stderr
execute_process(
    COMMAND ${TEST_EXECUTABLE}
    OUTPUT_FILE ${OUT_DIR}/${TEST_NAME}.out
    ERROR_FILE ${OUT_DIR}/${TEST_NAME}.stderr.out
    RESULT_VARIABLE test_result
)

# Return the exit code of the test
if(NOT test_result EQUAL 0)
    message(FATAL_ERROR "Test ${TEST_NAME} failed with exit code ${test_result}")
endif()
