# Add runtime library as a shared object
add_library(silly_runtime SHARED Silly_runtime.cpp)

set_target_properties(silly_runtime PROPERTIES OUTPUT_NAME "silly_runtime")

target_link_options(silly_runtime PRIVATE -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/version_script.map)

target_compile_options(silly_runtime PRIVATE ${COMMON_COMPILE_OPTIONS})

# Create symlink in build/lib/
add_custom_command(TARGET silly_runtime POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/lib
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        ../src/runtime/libsilly_runtime.so
        ${CMAKE_BINARY_DIR}/lib/libsilly_runtime.so
    COMMENT "Creating symlink: ${CMAKE_BINARY_DIR}/lib/libsilly_runtime.so -> src/runtime/libsilly_runtime.so"
)

if(SILLY_ENABLE_INSTALL)
    install(TARGETS silly_runtime
        LIBRARY DESTINATION lib
        NAMELINK_SKIP  # Don't install symlinks, just the actual .so
    )
endif()
