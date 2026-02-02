set(GRAMMAR_FILE "${CMAKE_CURRENT_SOURCE_DIR}/Silly.g4")

# ANTLR_OUTPUT_DIR is set in parent CMakeLists.txt

# Ensure output directory exists
file(MAKE_DIRECTORY ${ANTLR_OUTPUT_DIR})

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

add_custom_command(
    OUTPUT ${ANTLR_GENERATED_FILES}
    COMMAND antlr4 -Dlanguage=Cpp ${GRAMMAR_FILE} -listener -o ${ANTLR_OUTPUT_DIR}
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
