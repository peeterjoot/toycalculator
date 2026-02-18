# dwarfdump tests
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

    set(OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}")

    # Step 1: Run dwarfdump to create .dd file
    add_test(
        NAME EndToEnd.debug.${test_name}.dwarf_dump
        COMMAND bash -c "dwarfdump ${OUT_DIR}/${test_name}.o > ${OUT_DIR}/${test_name}.dd"
    )

    set_tests_properties(EndToEnd.debug.${test_name}.dwarf_dump
        PROPERTIES
            DEPENDS EndToEnd.debug.${test_name}
            LABELS "dwarf;debug"
    )

    # Step 2: Run dwarfcheck on the .dd file
    add_test(
        NAME EndToEnd.debug.${test_name}.dwarf_check
        COMMAND ${CMAKE_SOURCE_DIR}/bin/dwarfcheck
            --stem ${test_name}
            --dumppath ${OUT_DIR}/${test_name}.dd
    )

    set_tests_properties(EndToEnd.debug.${test_name}.dwarf_check
        PROPERTIES
            DEPENDS EndToEnd.debug.${test_name}.dwarf_dump
            LABELS "dwarf;debug"
    )
endforeach()
