# Create a top-level target for all endtoend tests
add_custom_target(endtoend_tests ALL)

function(add_endtoend_run_tests CATEGORY)
    cmake_parse_arguments(
        ARG
        "EXPECTED_ABORT"      # boolean flag
        "EXPECTED_EXIT_CODE;INPUT_FILE"  # single-value args
        "TESTS"               # multi-value args
        ${ARGN}
    )

    if(NOT DEFINED ARG_EXPECTED_EXIT_CODE)
        set(ARG_EXPECTED_EXIT_CODE 0)
    endif()

    foreach(source_file IN LISTS ARG_TESTS)
        get_filename_component(test_name ${source_file} NAME_WE)

        set(OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}")
        set(EXPECTED_DIR "${CMAKE_CURRENT_SOURCE_DIR}/expected")

        # Determine input file if specified
        if(DEFINED ARG_INPUT_FILE)
            set(INPUT_FILE_ARG "-DINPUT_FILE=${CMAKE_CURRENT_SOURCE_DIR}/${ARG_INPUT_FILE}")
        else()
            set(INPUT_FILE_ARG "")
        endif()

        # ─────────────────────────────────────────────────────
        # Test: Run the compiled binary and capture output
        # ─────────────────────────────────────────────────────
        add_test(
            NAME ${CATEGORY}.${test_name}
            COMMAND ${CMAKE_COMMAND}
                -DTEST_EXECUTABLE=${OUT_DIR}/${test_name}
                -DOUT_DIR=${OUT_DIR}
                -DTEST_NAME=${test_name}
                -DEXPECTED_EXIT_CODE=${ARG_EXPECTED_EXIT_CODE}
                -DEXPECTED_ABORT=${ARG_EXPECTED_ABORT}
                ${INPUT_FILE_ARG}
                -P ${CMAKE_SOURCE_DIR}/cmake/RunTest.cmake
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )

        set_tests_properties(${CATEGORY}.${test_name}
            PROPERTIES
                DEPENDS compile_${test_name}
                LABELS "run;end-to-end;compiler-cycle;sample;${CATEGORY}"
                TIMEOUT 60
        )

        # ─────────────────────────────────────────────────────
        # Test: Compare stdout
        # ────────────────────────────────────────────────────
        add_test(
            NAME ${CATEGORY}.${test_name}.compare_stdout
            COMMAND ${CMAKE_COMMAND} -E compare_files
                --ignore-eol
                ${EXPECTED_DIR}/${test_name}.out
                ${OUT_DIR}/${test_name}.out
        )

        set_tests_properties(${CATEGORY}.${test_name}.compare_stdout
            PROPERTIES
                DEPENDS ${CATEGORY}.${test_name}
                LABELS "compare;end-to-end;compiler-cycle;sample;${CATEGORY}"
        )

        # ─────────────────────────────────────────────────────
        # Test: Compare stderr (if expected file exists)
        # ─────────────────────────────────────────────────────
        if(EXISTS "${EXPECTED_DIR}/${test_name}.stderr.out")
            add_test(
                NAME ${CATEGORY}.${test_name}.compare_stderr
                COMMAND ${CMAKE_COMMAND} -E compare_files
                    --ignore-eol
                    ${EXPECTED_DIR}/${test_name}.stderr.out
                    ${OUT_DIR}/${test_name}.stderr.out
            )

            set_tests_properties(${CATEGORY}.${test_name}.compare_stderr
                PROPERTIES
                    DEPENDS ${CATEGORY}.${test_name}
                    LABELS "compare;end-to-end;compiler-cycle;sample;${CATEGORY}"
            )
        endif()
    endforeach()
endfunction()

function(add_endtoend_compile_tests)
    foreach(source_file IN LISTS ARGN)
        get_filename_component(test_name ${source_file} NAME_WE)

        set(OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}")
        set(SOURCE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/${source_file}")

        # Default flags
        set(COMPILER_FLAGS -g --emit-llvm --emit-mlir)

        # Allow per-test override via a variable named like COMPILER_FLAGS_${test_name}
        if(DEFINED COMPILER_FLAGS_${test_name})
            set(COMPILER_FLAGS ${COMPILER_FLAGS_${test_name}})
        endif()

        add_custom_command(
            OUTPUT
                "${OUT_DIR}/${test_name}"
                "${OUT_DIR}/${test_name}.mlir"
                "${OUT_DIR}/${test_name}.ll"
            COMMAND
                $<TARGET_FILE:silly>
                    --output-directory "${OUT_DIR}"
                    ${COMPILER_FLAGS}
                    "${SOURCE_PATH}"
            DEPENDS silly "${SOURCE_PATH}"
            COMMENT "Compiling ${test_name}"
            VERBATIM
        )

        # Create a target for this test so it gets built
        add_custom_target(compile_${test_name}
            DEPENDS "${OUT_DIR}/${test_name}"
        )

        # Add to the "all" target so it builds by default
        add_dependencies(endtoend_tests compile_${test_name})
    endforeach()
endfunction()

function( add_endtoend_multisource_compile_tests test_name )
    # ARGN contains all arguments after test_name — treat them as source files
    set( sources ${ARGN} )

    # Build full source paths for DEPENDS and compiler invocation
    set( source_paths )
    foreach( src ${sources} )
        list( APPEND source_paths "${CMAKE_CURRENT_SOURCE_DIR}/${src}" )
    endforeach()

    add_custom_command(
        OUTPUT
            "${CMAKE_CURRENT_BINARY_DIR}/${test_name}"
        COMMAND
            $<TARGET_FILE:silly>
            --output-directory "${CMAKE_CURRENT_BINARY_DIR}"
            -g
            -o ${test_name}
            ${source_paths}
        DEPENDS silly ${source_paths}
        COMMENT "Compiling ${test_name}"
        VERBATIM
    )

    add_custom_target( compile_${test_name}
        DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/${test_name}"
    )

    add_dependencies( endtoend_tests compile_${test_name} )
endfunction()

function( add_silly_compile_link_test test_name source )
    set( source_path "${CMAKE_CURRENT_SOURCE_DIR}/${source}" )
    get_filename_component( source_stem "${source}" NAME_WE )
    set( object_path "${CMAKE_CURRENT_BINARY_DIR}/${source_stem}.o" )
    set( exe_path "${CMAKE_CURRENT_BINARY_DIR}/${test_name}" )

    add_custom_command(
        OUTPUT
            "${exe_path}"
            "${object_path}"
        COMMAND
            $<TARGET_FILE:silly>
            --output-directory "${CMAKE_CURRENT_BINARY_DIR}"
            -g -c
            "${source_path}"
        COMMAND
            $<TARGET_FILE:silly>
            "${object_path}"
            -g
            -o "${exe_path}"
        DEPENDS silly "${source_path}"
        COMMENT "Compiling ${test_name}"
        VERBATIM
    )

    add_custom_target( compile_${test_name}
        DEPENDS "${exe_path}"
    )

    add_dependencies( endtoend_tests compile_${test_name} )
endfunction()

function(add_endtoend_failure_tests CATEGORY)
    set(TEST_FILES ${ARGN})

    foreach(source_file IN LISTS TEST_FILES)
        # Extract test name from filename
        get_filename_component(test_name ${source_file} NAME_WE)

        set(OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}")
        set(EXPECTED_DIR "${CMAKE_CURRENT_SOURCE_DIR}/expected")

        # Test: Compile should FAIL
        add_test(
            NAME ${CATEGORY}.${test_name}
            COMMAND ${CMAKE_COMMAND}
                -DSILLY_COMPILER=$<TARGET_FILE:silly>
                -DSOURCE_FILE=${CMAKE_CURRENT_SOURCE_DIR}/${source_file}
                -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
                -DOUT_DIR=${OUT_DIR}
                -DTEST_NAME=${test_name}
                -DEXPECTED_OUTPUT=${EXPECTED_DIR}/${test_name}.compile.out
                -P ${CMAKE_SOURCE_DIR}/cmake/RunFailureTest.cmake
        )

        set_tests_properties(${CATEGORY}.${test_name}
            PROPERTIES
                LABELS "failure;end-to-end;compiler-cycle;${CATEGORY}"
                TIMEOUT 30
        )
    endforeach()
endfunction()

# haven't generateed expected outputs for this one -- am not sure I should:
#add_subdirectory(sir)

add_subdirectory(driver)
add_subdirectory(debug)
add_subdirectory(exit)
add_subdirectory(failure)
add_subdirectory(fatal)
add_subdirectory(get)
add_subdirectory(mlirparsetest)
add_subdirectory(simple/array)
add_subdirectory(simple/bool)
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
