cmake_minimum_required(VERSION 3.16)
project(SillyBison)

find_package(BISON 3.2 REQUIRED)
find_package(FLEX REQUIRED)

BISON_TARGET(Parser
    ${CMAKE_CURRENT_SOURCE_DIR}/silly_bison.y
    ${CMAKE_CURRENT_BINARY_DIR}/silly_bison.tab.cc
    DEFINES_FILE ${CMAKE_CURRENT_BINARY_DIR}/silly_bison.tab.hh
    COMPILE_FLAGS "--language=c++"
)

FLEX_TARGET(Lexer
    ${CMAKE_CURRENT_SOURCE_DIR}/silly_bison.l
    ${CMAKE_CURRENT_BINARY_DIR}/silly_bison.lex.cc
    DEFINES_FILE ${CMAKE_CURRENT_BINARY_DIR}/silly_bison.lex.hh
)

ADD_FLEX_BISON_DEPENDENCY(Lexer Parser)

add_executable(silly_bison
    BisonParseListener.cpp
    ${BISON_Parser_OUTPUTS}
    ${FLEX_Lexer_OUTPUTS}
)

target_include_directories(silly_bison PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
)
