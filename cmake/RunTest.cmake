# This script runs a test executable and captures stdout/stderr

execute_process(
    COMMAND ${TEST_EXECUTABLE}
    OUTPUT_FILE ${OUT_DIR}/${TEST_NAME}.out
    ERROR_FILE ${OUT_DIR}/${TEST_NAME}.stderr.out
    RESULT_VARIABLE test_result
)

if(EXPECTED_ABORT)
    if(NOT test_result STREQUAL "Subprocess aborted")
        message(FATAL_ERROR
            "Test ${TEST_NAME} exited with code ${test_result}, "
            "expected abort")
    endif()
elseif(DEFINED EXPECTED_EXIT_CODE)
    if(NOT test_result EQUAL EXPECTED_EXIT_CODE)
        message(FATAL_ERROR
            "Test ${TEST_NAME} exited with code ${test_result}, "
            "expected ${EXPECTED_EXIT_CODE}")
    endif()
elseif(NOT test_result EQUAL 0)
    message(FATAL_ERROR
        "Test ${TEST_NAME} failed with exit code ${test_result}")
endif()
