/**
 * @file    silly.l
 * @author  Peeter Joot <peeterjoot@pm.me>
 * @brief   Bison based experimental parse tree listener and MLIR builder (Lexer part.)
 */
%option noyywrap nounput noinput
%option reentrant
%option bison-bridge
%option bison-locations
%option extra-type="silly::BisonParseListener*"

%{
#include "silly.tab.hh"
#include <climits>
#include <cerrno>
#include <cstring>
#include "BisonParseListener.hpp"

#define YY_USER_ACTION                         \
    yylloc->begin = yylloc->end;               \
    for ( const char* c = yytext; *c; ++c ) {  \
        if ( *c == '\n' ) {                    \
            yylloc->end.line++;                \
            yylloc->end.column = 1;            \
        } else {                               \
            yylloc->end.column++;              \
        }                                      \
    }
%}

%%

"PRINT"             { return silly::BisonParser::token::PRINT_TOKEN; }
";"                 { return silly::BisonParser::token::ENDOFSTATEMENT_TOKEN; }
"ABORT"             { return silly::BisonParser::token::ABORT_TOKEN; }
"]"                 { return silly::BisonParser::token::ARRAY_END_TOKEN; }
"["                 { return silly::BisonParser::token::ARRAY_START_TOKEN; }
"AND"               { return silly::BisonParser::token::BOOLEANAND_TOKEN; }
"OR"                { return silly::BisonParser::token::BOOLEANOR_TOKEN; }
"XOR"               { return silly::BisonParser::token::BOOLEANXOR_TOKEN; }
"BOOL"              { return silly::BisonParser::token::BOOL_TOKEN; }
")"                 { return silly::BisonParser::token::BRACE_END_TOKEN; }
"("                 { return silly::BisonParser::token::BRACE_START_TOKEN; }
"CALL"              { return silly::BisonParser::token::CALL_TOKEN; }
":"                 { return silly::BisonParser::token::COLON_TOKEN; }
","                 { return silly::BisonParser::token::COMMA_TOKEN; }
"CONTINUE"          { return silly::BisonParser::token::CONTINUE_TOKEN; }
"DECLARE"           { return silly::BisonParser::token::DECLARE_TOKEN; }
"/"                 { return silly::BisonParser::token::DIV_TOKEN; }
"ELIF"              { return silly::BisonParser::token::ELIF_TOKEN; }
"ELSE"              { return silly::BisonParser::token::ELSE_TOKEN; }
"EQ"                { return silly::BisonParser::token::EQUALITY_TOKEN; }
"="                 { return silly::BisonParser::token::EQUALS_TOKEN; }
"ERROR"             { return silly::BisonParser::token::ERROR_TOKEN; }
"EXIT"              { return silly::BisonParser::token::EXIT_TOKEN; }
"FLOAT32"           { return silly::BisonParser::token::FLOAT32_TOKEN; }
"FLOAT64"           { return silly::BisonParser::token::FLOAT64_TOKEN; }
"FOR"               { return silly::BisonParser::token::FOR_TOKEN; }
"FUNCTION"          { return silly::BisonParser::token::FUNCTION_TOKEN; }
"GET"               { return silly::BisonParser::token::GET_TOKEN; }
">="                { return silly::BisonParser::token::GREATEREQUAL_TOKEN; }
">"                 { return silly::BisonParser::token::GREATERTHAN_TOKEN; }
"IF"                { return silly::BisonParser::token::IF_TOKEN; }
"IMPORT"            { return silly::BisonParser::token::IMPORT_TOKEN; }
"INT16"             { return silly::BisonParser::token::INT16_TOKEN; }
"INT32"             { return silly::BisonParser::token::INT32_TOKEN; }
"INT64"             { return silly::BisonParser::token::INT64_TOKEN; }
"INT8"              { return silly::BisonParser::token::INT8_TOKEN; }
"{"                 { return silly::BisonParser::token::LEFT_CURLY_BRACKET_TOKEN; }
"<="                { return silly::BisonParser::token::LESSEQUAL_TOKEN; }
"<"                 { return silly::BisonParser::token::LESSTHAN_TOKEN; }
"MAIN"              { return silly::BisonParser::token::MAIN_TOKEN; }
"-"                 { return silly::BisonParser::token::MINUS_TOKEN; }
"%"                 { return silly::BisonParser::token::MOD_TOKEN; }
"MODULE"            { return silly::BisonParser::token::MODULE_TOKEN; }
"NE"                { return silly::BisonParser::token::NOTEQUAL_TOKEN; }
"NOT"               { return silly::BisonParser::token::NOT_TOKEN; }
"+"                 { return silly::BisonParser::token::PLUSCHAR_TOKEN; }
"RETURN"            { return silly::BisonParser::token::RETURN_TOKEN; }
"}"                 { return silly::BisonParser::token::RIGHT_CURLY_BRACKET_TOKEN; }
"STRING"            { return silly::BisonParser::token::STRING_TOKEN; }
"*"                 { return silly::BisonParser::token::TIMES_TOKEN; }

"TRUE" {
    return silly::BisonParser::token::TRUE_LITERAL;
}

"FALSE" {
    return silly::BisonParser::token::FALSE_LITERAL;
}

[0-9]+("."[0-9]+)?("E""-"?[0-9]+)?  {
    /* Float pattern: must be checked before INTEGER_PATTERN since it is a superset */
    if ( strchr( yytext, '.' ) || strchr( yytext, 'E' ) ) {
        yylval->emplace<std::string>( yytext );
        return silly::BisonParser::token::FLOAT_PATTERN;
    }
    yylval->emplace<std::string>( yytext );
    return silly::BisonParser::token::INTEGER_PATTERN;
}

\"[^\"]*\"  {
    /* String literal: not stripping the surrounding quotes -- leave that for createStringLiteral */
    yylval->emplace<std::string>( std::string( yytext, yyleng ) );
    return silly::BisonParser::token::STRING_PATTERN;
}

[a-zA-Z][a-zA-Z0-9_]*  {
    yylval->emplace<std::string>( yytext );
    return silly::BisonParser::token::IDENTIFIER;
}

[ \t\r\n]+          { /* skip whitespace */ }
"//"[^\n]*          { /* skip comments */ }
.                   { return yytext[0]; }

%%
