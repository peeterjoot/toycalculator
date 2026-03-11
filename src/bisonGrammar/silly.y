///
/// @file    silly.y
/// @author  Peeter Joot <peeterjoot@pm.me>
/// @brief   Bison based experimental parse tree listener and MLIR builder (Grammar part.)
///
%require "3.2"
%language "c++"
%defines "silly.tab.hh"
%define api.namespace {silly}
%define api.parser.class {BisonParser}
%define api.value.type variant
%define parse.error verbose
%locations

%code requires {
    #include <string>
    namespace silly {
        class BisonParseListener;

        struct Literal
        {
            enum class Kind
            {
                None,
                Int,
                Float,
                String,
                Bool
            };
            Kind kind{ Kind::None };
            std::string sval{}; /* int or float as string, or string content */
            bool bval{};

            static Literal makeNone()
            {
                return { Kind::None, {}, false };
            }
            static Literal makeInt( const std::string& s )
            {
                return { Kind::Int, s, false };
            }
            static Literal makeFloat( const std::string& s )
            {
                return { Kind::Float, s, false };
            }
            static Literal makeString( const std::string& s )
            {
                return { Kind::String, s, false };
            }
            static Literal makeBool( bool b )
            {
                return { Kind::Bool, {}, b };
            }
        };
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
    #include "silly.lex.hh"
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
%token DECLARE_TOKEN
%token DIV_TOKEN
%token ELIF_TOKEN
%token ELSE_TOKEN
%token ENDOFSTATEMENT_TOKEN
%token EQUALITY_TOKEN
%token EQUALS_TOKEN
%token ERROR_TOKEN
%token EXIT_TOKEN
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

/* Typed tokens */
%token <std::string> INTEGER_PATTERN
%token <std::string> FLOAT_PATTERN
%token <std::string> STRING_PATTERN
%token <std::string> IDENTIFIER

/* Typed non-terminals */
%type <silly::Literal> literal
%type <std::string>    integerLiteral
%type <std::string>    floatLiteral
%type <std::string>    stringLiteral
%type <std::string>    intType
%type <std::string>    floatType
%type <std::string>    arrayBoundsExpression
%type <silly::Literal> optionalInitializer

%%

startRule
    : statementList
        { driver.exitStartRule( @$ ); }
    ;

statementList
    : /* empty */
        { driver.enterStartRule( @$ ); }
    | statementList statement
    ;

statement
    : printStatement ENDOFSTATEMENT_TOKEN
    | declareStatement ENDOFSTATEMENT_TOKEN
    ;

declareStatement
    : intDeclareStatement
    | floatDeclareStatement
    | boolDeclareStatement
    ;

intDeclareStatement
    : intType IDENTIFIER arrayBoundsExpression optionalInitializer
        { driver.enterIntDeclare( $1, $2, $3, $4, @1, @2, @3 ); }
    ;

floatDeclareStatement
    : floatType IDENTIFIER arrayBoundsExpression optionalInitializer
        { driver.enterFloatDeclare( $1, $2, $3, $4, @1, @2, @3 ); }
    ;

boolDeclareStatement
    : BOOL_TOKEN IDENTIFIER arrayBoundsExpression optionalInitializer
        { driver.enterBoolDeclare( $2, $3, $4, @1, @2, @3 ); }
    ;

intType
    : INT8_TOKEN    { $$ = "INT8";  }
    | INT16_TOKEN   { $$ = "INT16"; }
    | INT32_TOKEN   { $$ = "INT32"; }
    | INT64_TOKEN   { $$ = "INT64"; }
    ;

floatType
    : FLOAT32_TOKEN { $$ = "FLOAT32"; }
    | FLOAT64_TOKEN { $$ = "FLOAT64"; }
    ;

/* Returns -1 if no array bounds, otherwise the array size */
arrayBoundsExpression
    : /* empty */
        { $$ = {}; }
    | ARRAY_START_TOKEN INTEGER_PATTERN ARRAY_END_TOKEN
        { $$ = $2; }
    ;

/* Returns a Literal with kind==Bool/bval==false as the "no initializer" sentinel */
optionalInitializer
    : /* empty */
        { $$ = silly::Literal::makeNone(); }
    | EQUALS_TOKEN literal
        { $$ = $2; }
    | LEFT_CURLY_BRACKET_TOKEN literal RIGHT_CURLY_BRACKET_TOKEN
        { $$ = $2; }
    ;

printStatement
    : PRINT_TOKEN literal
        { driver.enterPrintStatement( $2, @1, @2 ); }
    ;

literal
    : integerLiteral
        { $$ = silly::Literal::makeInt( $1 ); }
    | floatLiteral
        { $$ = silly::Literal::makeFloat( $1 ); }
    | stringLiteral
        { $$ = silly::Literal::makeString( $1 ); }
    | TRUE_LITERAL
        { $$ = silly::Literal::makeBool( true ); }
    | FALSE_LITERAL
        { $$ = silly::Literal::makeBool( false ); }
    ;

integerLiteral
    : INTEGER_PATTERN
        { $$ = $1; }
    ;

floatLiteral
    : FLOAT_PATTERN
        { $$ = $1; }
    ;

stringLiteral
    : STRING_PATTERN
        { $$ = $1; }
    ;

%%

void silly::BisonParser::error( const location_type& loc, const std::string& msg )
{
    driver.emitParseError( loc, msg );
}
