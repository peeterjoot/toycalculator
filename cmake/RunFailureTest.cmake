# This script runs the compiler on a file expected to fail,
# captures the error output, and compares it to expected output.

# Extract just the filename from the full path
get_filename_component(SOURCE_FILENAME ${SOURCE_FILE} NAME)

execute_process(
    COMMAND ${SILLY_COMPILER} ${COMPILER_FLAGS} ${SOURCE_FILENAME}
        --output-directory ${OUT_DIR}
    OUTPUT_VARIABLE compile_stdout
    ERROR_VARIABLE compile_stderr
    RESULT_VARIABLE compile_result
    WORKING_DIRECTORY ${SOURCE_DIR}  # Run from source directory
)

# Compilation should have FAILED (non-zero exit code)
if(compile_result EQUAL 0)
    message(FATAL_ERROR "Expected compilation to fail, but it succeeded!")
endif()

# Combine stdout and stderr (silly outputs errors to stderr)
set(actual_output "${compile_stderr}")

# Read expected output
if(EXISTS ${EXPECTED_OUTPUT})
    file(READ ${EXPECTED_OUTPUT} expected_output)

    # Normalize line endings
    string(REGEX REPLACE "\r\n" "\n" actual_output "${actual_output}")
    string(REGEX REPLACE "\r\n" "\n" expected_output "${expected_output}")

    # Compare outputs
    if(NOT actual_output STREQUAL expected_output)
        message("Expected output:")
        message("${expected_output}")
        message("\nActual output:")
        message("${actual_output}")
        message(FATAL_ERROR "Compiler error output does not match expected output")
    endif()
else()
    # No expected file - just verify it failed
    message("Note: No expected output file at ${EXPECTED_OUTPUT}")
    message("Compilation failed as expected with output:")
    message("${actual_output}")
endif()
