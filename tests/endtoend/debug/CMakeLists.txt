# dwarfdump tests
if(FILECHECK_EXECUTABLE)
    set(DWARFDUMP_TESTS
        for_simplest.silly
        foo.silly
        f.silly
    )

    add_endtoend_compile_tests(${DWARFDUMP_TESTS})

    add_endtoend_run_tests("EndToEnd.debug"
        EXPECTED_EXIT_CODE 0
        TESTS ${DWARFDUMP_TESTS}
    )

    foreach(source_file IN LISTS DWARFDUMP_TESTS)
        get_filename_component(test_name ${source_file} NAME_WE)

        # Run dwarfdump (saving to a .dd suffixed file)
        add_test(
            NAME EndToEnd.debug.${test_name}.dwarf_dump
            COMMAND bash -c "dwarfdump ${CMAKE_CURRENT_BINARY_DIR}/${test_name}.o > ${CMAKE_CURRENT_BINARY_DIR}/${test_name}.dd"
        )

        set_tests_properties(EndToEnd.debug.${test_name}.dwarf_dump
            PROPERTIES
                DEPENDS EndToEnd.debug.${test_name}
                LABELS "dwarf;debug"
        )

        # check that dump against the FileCheck patterns:
        add_test(
            NAME EndToEnd.debug.${test_name}.dwarf_check
            COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_CURRENT_BINARY_DIR}
                ${FILECHECK_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${test_name}.check --input-file=${test_name}.dd
        )

        set_tests_properties(EndToEnd.debug.${test_name}.dwarf_check
            PROPERTIES
                DEPENDS EndToEnd.debug.${test_name}.dwarf_dump
                LABELS "dwarf;debug;filecheck"
        )
    endforeach()
endif()
