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

# Add subdirectories
add_subdirectory(array)
add_subdirectory(bool)
add_subdirectory(builtins)
add_subdirectory(debug)
add_subdirectory(exit)
add_subdirectory(expressions)
add_subdirectory(failure)
add_subdirectory(for)
add_subdirectory(function)
add_subdirectory(ifelifelse)
add_subdirectory(init)
add_subdirectory(misc)
add_subdirectory(operators)
add_subdirectory(print)
add_subdirectory(simple)
add_subdirectory(string)
add_subdirectory(types)
