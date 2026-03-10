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

%token PRINT_TOKEN ENDOFSTATEMENT_TOKEN
%token <int> INTEGER_PATTERN

%%

startRule
    : statementList
    ;

statementList
    : /* empty */
    | statementList statement
    ;

statement
    : printStatement ENDOFSTATEMENT_TOKEN
    ;

printStatement
    : PRINT_TOKEN INTEGER_PATTERN
        { driver.emitPrint( $2, @2 ); }
    ;

%%

void silly::BisonParser::error( const location_type& loc, const std::string& msg )
{
    driver.emitError( loc, msg );
}
