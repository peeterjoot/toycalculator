BISON_TARGET(Parser
    ${CMAKE_CURRENT_SOURCE_DIR}/silly.y
    ${CMAKE_CURRENT_BINARY_DIR}/silly.tab.cc
    DEFINES_FILE ${CMAKE_CURRENT_BINARY_DIR}/silly.tab.hh
    COMPILE_FLAGS "--language=c++"
)

FLEX_TARGET(Lexer
    ${CMAKE_CURRENT_SOURCE_DIR}/silly.l
    ${CMAKE_CURRENT_BINARY_DIR}/silly.lex.cc
    DEFINES_FILE ${CMAKE_CURRENT_BINARY_DIR}/silly.lex.hh
)

ADD_FLEX_BISON_DEPENDENCY(Lexer Parser)

add_library(BisonGrammar
    BisonParseListener.cpp
    ${BISON_Parser_OUTPUTS}
    ${FLEX_Lexer_OUTPUTS}
)

target_include_directories(BisonGrammar PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}   # for generated .tab.hh and .lex.hh
    ${CMAKE_CURRENT_SOURCE_DIR}   # for BisonParseListener.hpp etc.
)
