///
/// @file    silly_bison.l
/// @author  Peeter Joot <peeterjoot@pm.me>
/// @brief   Bison based experimental parse tree listener and MLIR builder (Grammar part.)
///
%require "3.2"
%language "c++"
%defines "silly_bison.tab.hh"

%define api.namespace {silly}
%define api.parser.class {BisonParser}
%define api.value.type variant
%define parse.error verbose
%locations

%code requires {
    #include <string>
    namespace silly {
        class BisonParseListener;
    }
}

%code provides {
    #ifndef YYSTYPE
    # define YYSTYPE silly::BisonParser::semantic_type
    #endif
    #ifndef YYLTYPE
    # define YYLTYPE silly::BisonParser::location_type
    #endif
}

%code top {
    #include "BisonParseListener.hpp"
    #include "silly_bison.lex.hh"

    static int yylex( silly::BisonParser::value_type* yylval,
                      silly::BisonParser::location_type* yylloc,
                      silly::BisonParseListener& driver )
    {
        return ::yylex( yylval, yylloc, driver.getScanner() );
    }
}

%param { silly::BisonParseListener& driver }

%token ABORT_TOKEN
%token ARRAY_END_TOKEN
%token ARRAY_START_TOKEN
%token BOOLEANAND_TOKEN
%token BOOLEANOR_TOKEN
%token BOOLEANXOR_TOKEN
%token BOOL_TOKEN
%token BRACE_END_TOKEN
%token BRACE_START_TOKEN
%token CALL_TOKEN
%token COLON_TOKEN
%token COMMA_TOKEN
%token CONTINUE_TOKEN
%token DECIMALSEP_TOKEN
%token DECLARE_TOKEN
%token DIV_TOKEN
%token DQUOTE_TOKEN
%token ELIF_TOKEN
%token ELSE_TOKEN
%token ENDOFSTATEMENT_TOKEN
%token EQUALITY_TOKEN
%token EQUALS_TOKEN
%token ERROR_TOKEN
%token EXIT_TOKEN
%token EXPONENT_TOKEN
%token FALSE_LITERAL
%token FLOAT32_TOKEN
%token FLOAT64_TOKEN
%token FOR_TOKEN
%token FUNCTION_TOKEN
%token GET_TOKEN
%token GREATEREQUAL_TOKEN
%token GREATERTHAN_TOKEN
%token IF_TOKEN
%token IMPORT_TOKEN
%token INT16_TOKEN
%token INT32_TOKEN
%token INT64_TOKEN
%token INT8_TOKEN
%token LEFT_CURLY_BRACKET_TOKEN
%token LESSEQUAL_TOKEN
%token LESSTHAN_TOKEN
%token MAIN_TOKEN
%token MINUS_TOKEN
%token MOD_TOKEN
%token MODULE_TOKEN
%token NOTEQUAL_TOKEN
%token NOT_TOKEN
%token PLUSCHAR_TOKEN
%token PRINT_TOKEN
%token RETURN_TOKEN
%token RIGHT_CURLY_BRACKET_TOKEN
%token STRING_TOKEN
%token TIMES_TOKEN
%token TRUE_LITERAL
%token <int> INTEGER_PATTERN

%%

startRule
    : statementList
        { driver.exit( @$ ); }
    ;

statementList
    :
    /* empty */
        { driver.enter( @$ ); }
    | statementList statement
    ;

statement
    : printStatement ENDOFSTATEMENT_TOKEN
    ;

printStatement
    : PRINT_TOKEN INTEGER_PATTERN
        { driver.emitPrint( $2, @1, @2 ); }
    ;

%%

void silly::BisonParser::error( const location_type& loc, const std::string& msg )
{
    driver.emitError( loc, msg );
}
