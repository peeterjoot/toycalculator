# Define the helper function
function(add_endtoend_tests TEST_CATEGORY)
    # Parse the test files from the remaining arguments
    set(TEST_FILES ${ARGN})

    foreach(test_file IN LISTS TEST_FILES)
        get_filename_component(test_name ${test_file} NAME_WE)

        add_test(
            NAME ${TEST_CATEGORY}.${test_name}
            COMMAND ${CMAKE_SOURCE_DIR}/bin/testit
                --driverpath ${CMAKE_BINARY_DIR}/bin
                --just ${test_file}
                --out ${CMAKE_BINARY_DIR}/out/${TEST_CATEGORY}/${test_name}
                --clean
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )

        set_tests_properties(${TEST_CATEGORY}.${test_name}
            PROPERTIES
                LABELS "end-to-end;compiler-cycle;sample;${TEST_CATEGORY}"
                TIMEOUT 60
                PASS_REGULAR_EXPRESSION "TEST WAS SUCCESSFUL"
                ENVIRONMENT "CTEST_OUTPUT_ON_FAILURE=1"
        )
    endforeach()
endfunction()

function(add_simple_endtoend_test TEST_NAME SOURCE_FILE)
    set(OUT_DIR "${CMAKE_BINARY_DIR}/out/endtoend/${TEST_NAME}")
    set(EXPECTED_STDOUT "${CMAKE_CURRENT_SOURCE_DIR}/expected/${TEST_NAME}.out")
    set(EXPECTED_STDERR "${CMAKE_CURRENT_SOURCE_DIR}/expected/${TEST_NAME}.stderr.out")

    # ───────────────────────────────────────────────
    # Step 1: compile
    # ───────────────────────────────────────────────
    add_custom_command(
        OUTPUT
            "${OUT_DIR}/${TEST_NAME}"
            "${OUT_DIR}/${TEST_NAME}.mlir"
            "${OUT_DIR}/${TEST_NAME}.ll"
            "${OUT_DIR}/${TEST_NAME}.compile.out"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${OUT_DIR}"
        COMMAND
            $<TARGET_FILE:silly>
                --output-directory "${OUT_DIR}"
                -g --emit-llvm --emit-mlir
                "${SOURCE_FILE}"
            > "${OUT_DIR}/${TEST_NAME}.compile.out" 2>&1
        DEPENDS silly ${SOURCE_FILE}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Compiling ${TEST_NAME}"
    )

    # ───────────────────────────────────────────────
    # Step 2: run binary
    # ───────────────────────────────────────────────
    add_custom_command(
        OUTPUT "${OUT_DIR}/${TEST_NAME}.out" "${OUT_DIR}/${TEST_NAME}.stderr.out"
        COMMAND
            "${OUT_DIR}/${TEST_NAME}"
            > "${OUT_DIR}/${TEST_NAME}.out" 2> "${OUT_DIR}/${TEST_NAME}.stderr.out"
        DEPENDS "${OUT_DIR}/${TEST_NAME}"
        COMMENT "Running ${TEST_NAME}"
    )

    # ──────────────────────────────────────────────
    # Step 3: compare outputs
    # ───────────────────────────────────────────────
    add_test(
        NAME endtoend.${TEST_NAME}
        COMMAND ${CMAKE_COMMAND} -E compare_files
            --ignore-eol
            "${EXPECTED_STDOUT}" "${OUT_DIR}/${TEST_NAME}.out"
    )

    if(EXISTS "${EXPECTED_STDERR}")
        add_test(
            NAME endtoend.${TEST_NAME}.stderr
            COMMAND ${CMAKE_COMMAND} -E compare_files
                --ignore-eol
                "${EXPECTED_STDERR}" "${OUT_DIR}/${TEST_NAME}.stderr.out"
        )
        set_tests_properties(endtoend.${TEST_NAME}.stderr
            PROPERTIES DEPENDS endtoend.${TEST_NAME}
        )
    endif()

    set_tests_properties(endtoend.${TEST_NAME}
        PROPERTIES
            LABELS "end-to-end;compiler-cycle;sample"
            TIMEOUT 60
            REQUIRED_FILES "${OUT_DIR}/${TEST_NAME}.out"
            ENVIRONMENT "CTEST_OUTPUT_ON_FAILURE=1"
    )
endfunction()

# ------------------------------------------------------------------------------
# add_endtoend_tests_simple
#
# Convenience wrapper that calls add_simple_endtoend_test for each test name
# in the provided list.
#
# Usage:
#   add_endtoend_tests_simple(EndToEnd.for
#       for_complex_body
#       for
#       for_simplest
#       forward_ref_init
#       negative_step_for
#       nested_for
#       zero_iter_for
#       induction_not_used
#   )
#
# Assumptions:
#   - Source files are named <test_name>.silly in ${CMAKE_CURRENT_SOURCE_DIR}
#   - Expected outputs are in expected/<test_name>.out (and optionally .stderr.out)
#
# ------------------------------------------------------------------------------
function(add_endtoend_tests_simple CATEGORY)
    set(TEST_NAMES ${ARGN})

    foreach(test_name IN LISTS TEST_NAMES)
        set(source_file "${test_name}.silly")
        add_simple_endtoend_test("${test_name}" "${source_file}")

        # Optional: tag the test with the category for easier filtering
        set_tests_properties(endtoend.${test_name}
            PROPERTIES
                LABELS "end-to-end;compiler-cycle;sample;${CATEGORY}"
        )
    endforeach()
endfunction()

add_subdirectory(complex/array)
add_subdirectory(complex/bool)
add_subdirectory(complex/builtins)
add_subdirectory(complex/exit)
add_subdirectory(complex/failure)
add_subdirectory(complex/init)
add_subdirectory(complex/types)
add_subdirectory(simple/array)
add_subdirectory(simple/bool)
add_subdirectory(simple/debug)
add_subdirectory(simple/expressions)
add_subdirectory(simple/for)
add_subdirectory(simple/function)
add_subdirectory(simple/ifelifelse)
add_subdirectory(simple/init)
add_subdirectory(simple/misc)
add_subdirectory(simple/operators)
add_subdirectory(simple/print)
add_subdirectory(simple/simple)
add_subdirectory(simple/string)
add_subdirectory(simple/types)
