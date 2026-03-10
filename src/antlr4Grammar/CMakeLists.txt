set(GRAMMAR_FILE "${CMAKE_CURRENT_SOURCE_DIR}/Silly.g4")

# ANTLR_OUTPUT_DIR is set in parent CMakeLists.txt

# Ensure output directory exists
file(MAKE_DIRECTORY ${ANTLR_OUTPUT_DIR})

# Define the ANTLR generated source files
set(ANTLR_SOURCES
  ${ANTLR_OUTPUT_DIR}/SillyBaseListener.cpp
  ${ANTLR_OUTPUT_DIR}/SillyListener.cpp
  ${ANTLR_OUTPUT_DIR}/SillyParser.cpp
  ${ANTLR_OUTPUT_DIR}/SillyLexer.cpp
)

# List all generated files (including headers!)
set(ANTLR_GENERATED_FILES
    ${ANTLR_OUTPUT_DIR}/SillyBaseListener.cpp
    ${ANTLR_OUTPUT_DIR}/SillyBaseListener.h
    ${ANTLR_OUTPUT_DIR}/SillyListener.cpp
    ${ANTLR_OUTPUT_DIR}/SillyListener.h
    ${ANTLR_OUTPUT_DIR}/SillyParser.cpp
    ${ANTLR_OUTPUT_DIR}/SillyParser.h
    ${ANTLR_OUTPUT_DIR}/SillyLexer.cpp
    ${ANTLR_OUTPUT_DIR}/SillyLexer.h
)

# ------------------------------------------------------------------------------
# Locate ANTLR tool: prefer local JAR in source tree (if found: for WSL which has mismatched antlr4 vs. includes), else system antlr4
# ------------------------------------------------------------------------------

set(ANTLR_JAR "${CMAKE_SOURCE_DIR}/antlr-4.10-complete.jar")

if(EXISTS "${ANTLR_JAR}")
    # Ensure we have java at all (for JAR case)
    find_package(Java COMPONENTS Runtime REQUIRED)

    # Use java -jar <local-jar> ...
    set(ANTLR_COMMAND
        ${Java_JAVA_EXECUTABLE} -jar "${ANTLR_JAR}"
    )
    message(STATUS "Using local ANTLR JAR: ${ANTLR_JAR}")
else()
    # Fall back to whatever 'antlr4' resolves to (system wrapper, alias, etc.)
    find_program(ANTLR4_EXECUTABLE NAMES antlr4 antlr)
    if(NOT ANTLR4_EXECUTABLE)
        message(FATAL_ERROR "ANTLR tool not found: neither ${ANTLR_JAR} exists nor 'antlr4' is in PATH")
    endif()
    set(ANTLR_COMMAND "${ANTLR4_EXECUTABLE}")
    message(STATUS "Using system ANTLR: ${ANTLR4_EXECUTABLE}")
endif()

# ------------------------------------------------------------------------------

add_custom_command(
    OUTPUT ${ANTLR_GENERATED_FILES}
    COMMAND ${ANTLR_COMMAND} -Dlanguage=Cpp ${GRAMMAR_FILE} -listener -o ${ANTLR_OUTPUT_DIR}
    DEPENDS ${GRAMMAR_FILE}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating ANTLR grammar files from Silly.g4"
    VERBATIM
)

# Create the target that driver depends on
add_custom_target(GenerateANTLR
    DEPENDS ${ANTLR_GENERATED_FILES}
)

# Mark files as generated
set_source_files_properties(
    ${ANTLR_GENERATED_FILES}
    PROPERTIES GENERATED TRUE
)

add_library(Antlr4Grammar
    ${ANTLR_SOURCES}
    Antlr4ParseListener.cpp
)

target_include_directories(Antlr4Grammar PRIVATE ${ANTLR_OUTPUT_DIR})

add_dependencies(Antlr4Grammar GenerateANTLR)
