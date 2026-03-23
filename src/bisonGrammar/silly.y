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
    #include <vector>
    #include <memory>
    #include "location.hh"   /* for silly::location */

    namespace silly {
        class BisonParseListener;

        enum class ExprOp : uint32_t
        {
            None,
            Mul,
            Div,
            Mod,
            Add,
            Sub,
            Or,
            And,
            Xor,
            Less,
            LessEqual,
            Greater,
            GreaterEqual,
            Equal,
            NotEqual,
            Minus,
            Plus,
            Not,
        };

        enum class Types : uint32_t
        {
            None,
            Boolean,
            Int8,
            Int16,
            Int32,
            Int64,
            Float32,
            Float64
        };

        struct TypeAndName
        {
            Types typ{};
            std::string name{};
            silly::location loc{};
        };

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

        struct Expr
        {
            enum class Kind
            {
                Literal,
                Variable,
                ArrayVariable,
                UnaryOp,
                BinaryOp,
                Call,
            };

            Kind kind;
            Literal lit{};
            std::string name{};
            ExprOp op{ExprOp::None};
            std::shared_ptr<Expr> left{};  /* for UnaryOp: operand; BinaryOp: lhs, also index expressions */
            std::shared_ptr<Expr> right{}; /* for BinaryOp: rhs */

            static Expr fromLiteral( const Literal& l )
            {
                return { Kind::Literal, l };
            }

            static Expr fromVariable( const std::string& s )
            {
                return { Kind::Variable, {}, s };
            }

            static Expr fromArrayVariable( const std::string& s, const Expr& idx )
            {
                return { Kind::ArrayVariable, {}, s, ExprOp::None, std::make_shared<Expr>( idx ) };
            }

            static Expr makeUnaryOp( const ExprOp & op, const Expr& operand )
            {
                return { Kind::UnaryOp, {}, {}, op, std::make_shared<Expr>( operand ) };
            }

            static Expr makeBinaryOp( const ExprOp& op, const Expr& l, const Expr& r )
            {
                return { Kind::BinaryOp, {}, {}, op, std::make_shared<Expr>( l ), std::make_shared<Expr>( r ) };
            }

            static Expr makeCall( const std::string& name )
            {
                return { Kind::Call, {}, name };
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
%type <silly::Literal>                      literal
%type <std::string>                         integerLiteral
%type <std::string>                         floatLiteral
%type <std::string>                         stringLiteral
%type <silly::Types>                        boolType
%type <silly::Types>                        intType
%type <silly::Types>                        floatType
%type <std::string>                         arrayBoundsExpression
%type <std::vector<silly::Literal>>         optionalInitializer
%type <std::vector<silly::Literal>>         initializerList
%type <std::string>                         importStatement
%type <silly::Types>                        scalarType
%type <std::vector<silly::Expr>>            printArgList
%type <silly::Expr>                         assignmentLHS
%type <silly::Expr>                         expression
%type <silly::Types>                        optionalReturnType
%type <std::vector<silly::TypeAndName>>     paramList
%type <std::vector<silly::TypeAndName>>     optionalParamList
%type <silly::TypeAndName>                  variableTypeAndName

%left BOOLEANOR_TOKEN
%left BOOLEANXOR_TOKEN
%left BOOLEANAND_TOKEN
%nonassoc EQUALITY_TOKEN NOTEQUAL_TOKEN
%nonassoc LESSTHAN_TOKEN LESSEQUAL_TOKEN GREATERTHAN_TOKEN GREATEREQUAL_TOKEN
%left PLUSCHAR_TOKEN MINUS_TOKEN
%left TIMES_TOKEN DIV_TOKEN MOD_TOKEN
%right UMINUS UPLUS UNOT       /* pseudo-tokens for unary precedence */

%%

startRule
    : moduleProgram
        { driver.exitStartRule( @$ ); }
    | mainProgram
        { driver.exitStartRule( @$ ); }
    ;

moduleProgram
    : MODULE_TOKEN ENDOFSTATEMENT_TOKEN
        { driver.setModule(); driver.enterStartRule( @$ ); }
      moduleStatementList
    ;

moduleStatementList
    : /* empty */
    | moduleStatementList moduleStatement ENDOFSTATEMENT_TOKEN
    ;

moduleStatement
    : functionStatement
    | importStatement
    ;

mainProgram
    : MAIN_TOKEN ENDOFSTATEMENT_TOKEN
        { driver.enterStartRule( @$ ); }
      statementList optionalExitStatement
    | /* no MAIN token */
        { driver.enterStartRule( @$ ); }
      statementList optionalExitStatement
    ;

optionalExitStatement
    : /* empty */
    | exitStatement ENDOFSTATEMENT_TOKEN
    ;

exitStatement
    : EXIT_TOKEN expression
        { driver.enterExitStatement( @2, $2 ); }
    | EXIT_TOKEN
        { driver.enterExitStatement( @1 ); }
    ;

statementList
    : /* empty */
    | statementList statement
    ;

statement
    : abortStatement ENDOFSTATEMENT_TOKEN
    | assignmentStatement ENDOFSTATEMENT_TOKEN
    | callStatement ENDOFSTATEMENT_TOKEN
    | declareStatement ENDOFSTATEMENT_TOKEN
    | errorStatement ENDOFSTATEMENT_TOKEN
    | forStatement ENDOFSTATEMENT_TOKEN
    | functionStatement ENDOFSTATEMENT_TOKEN
    | getStatement ENDOFSTATEMENT_TOKEN
    | ifElifElseStatement ENDOFSTATEMENT_TOKEN
    | importStatement ENDOFSTATEMENT_TOKEN
    | printStatement ENDOFSTATEMENT_TOKEN
    ;

abortStatement
    : ABORT_TOKEN
        { driver.enterAbortStatement( @1 ); }
    ;

callStatement
    : CALL_TOKEN IDENTIFIER BRACE_START_TOKEN optionalCallArgList BRACE_END_TOKEN
        { /* stub */ }
    ;

optionalCallArgList
    : /* empty */
    | callArgList
    ;

callArgList
    : expression
    | callArgList COMMA_TOKEN expression
    ;

getStatement
    : GET_TOKEN IDENTIFIER
        { driver.enterGetStatement( @2, $2 ); }
    | GET_TOKEN IDENTIFIER ARRAY_START_TOKEN expression ARRAY_END_TOKEN
        { driver.enterGetStatement( @2, $2, $4 ); }
    ;

forStatement
    : FOR_TOKEN
      BRACE_START_TOKEN intType IDENTIFIER COLON_TOKEN
          BRACE_START_TOKEN
              expression COMMA_TOKEN expression
          BRACE_END_TOKEN
      BRACE_END_TOKEN
      scopedStatements
        { /* stub */ }
    | FOR_TOKEN
      BRACE_START_TOKEN intType IDENTIFIER COLON_TOKEN
          BRACE_START_TOKEN
              expression COMMA_TOKEN expression COMMA_TOKEN expression
          BRACE_END_TOKEN
      BRACE_END_TOKEN
      scopedStatements
        { /* stub */ }
    ;

ifElifElseStatement
    : ifStatement elifStatementList optionalElseStatement
        { /* stub */ }
    ;

ifStatement
    : IF_TOKEN BRACE_START_TOKEN expression BRACE_END_TOKEN scopedStatements
        { /* stub */ }
    ;

elifStatementList
    : /* empty */
    | elifStatementList elifStatement
    ;

elifStatement
    : ELIF_TOKEN BRACE_START_TOKEN expression BRACE_END_TOKEN scopedStatements
        { /* stub */ }
    ;

optionalElseStatement
    : /* empty */
    | ELSE_TOKEN scopedStatements
        { /* stub */ }
    ;

scopedStatements
    : LEFT_CURLY_BRACKET_TOKEN statementList optionalReturnStatement RIGHT_CURLY_BRACKET_TOKEN
        { /* stub */ }
    ;

optionalReturnStatement
    : /* empty */
    | RETURN_TOKEN expression ENDOFSTATEMENT_TOKEN
        { /* stub */ }
    | RETURN_TOKEN ENDOFSTATEMENT_TOKEN
        { /* stub */ }
    ;

assignmentStatement
    : assignmentLHS EQUALS_TOKEN expression
        { driver.enterAssignmentStatement( $1, $3, @1, @3 ); }
    ;

assignmentLHS
    : IDENTIFIER
        { $$ = silly::Expr::fromVariable( $1 ); }
    | IDENTIFIER ARRAY_START_TOKEN expression ARRAY_END_TOKEN
        { $$ = silly::Expr::fromArrayVariable( $1, $3 ); }
    ;

importStatement
    : IMPORT_TOKEN IDENTIFIER
        { driver.enterImportStatement( @2, $2 ); }
    ;

functionStatement
    : FUNCTION_TOKEN IDENTIFIER
      BRACE_START_TOKEN optionalParamList BRACE_END_TOKEN
      optionalReturnType
        { driver.enterFunctionPrototype( $2, $4, $6, @1, @2 ); }
    | FUNCTION_TOKEN IDENTIFIER
      BRACE_START_TOKEN optionalParamList BRACE_END_TOKEN
      optionalReturnType
      scopedStatements
        { driver.enterFunctionDefinition( $2, $4, $6, @1, @2 ); }
    ;

optionalParamList
    : /* empty */
        { $$ = {}; }
    | paramList
        { $$ = std::move( $1 ); }
    ;

paramList
    : variableTypeAndName
        { $$ = std::vector<silly::TypeAndName>{ $1 }; }
    | paramList COMMA_TOKEN variableTypeAndName
        { $1.push_back( $3 ); $$ = std::move( $1 ); }
    ;

variableTypeAndName
    : scalarType IDENTIFIER
        { $$ = { $1, $2, @2 }; }
    ;

optionalReturnType
    : /* empty */
        { $$ = silly::Types::None; }
    | COLON_TOKEN scalarType
        { $$ = $2; }
    ;

scalarType
    : intType       { $$ = $1; }
    | floatType     { $$ = $1; }
    | boolType      { $$ = $1; }
    ;

declareStatement
    : intDeclareStatement
    | floatDeclareStatement
    | boolDeclareStatement
    ;

intDeclareStatement
    : intType IDENTIFIER arrayBoundsExpression optionalInitializer
        { driver.enterIntDeclareStatement( $1, $2, $3, $4, @1, @2, @3 ); }
    ;

floatDeclareStatement
    : floatType IDENTIFIER arrayBoundsExpression optionalInitializer
        { driver.enterFloatDeclareStatement( $1, $2, $3, $4, @1, @2, @3 ); }
    ;

boolDeclareStatement
    : boolType IDENTIFIER arrayBoundsExpression optionalInitializer
        { driver.enterBoolDeclareStatement( $2, $3, $4, @1, @2, @3 ); }
    ;

intType
    : INT8_TOKEN    { $$ = silly::Types::Int8;  }
    | INT16_TOKEN   { $$ = silly::Types::Int16; }
    | INT32_TOKEN   { $$ = silly::Types::Int32; }
    | INT64_TOKEN   { $$ = silly::Types::Int64; }
    ;

boolType
    : BOOL_TOKEN    { $$ = silly::Types::Boolean;  }
    ;

floatType
    : FLOAT32_TOKEN { $$ = silly::Types::Float32; }
    | FLOAT64_TOKEN { $$ = silly::Types::Float64; }
    ;

/* Returns -1 if no array bounds, otherwise the array size */
arrayBoundsExpression
    : /* empty */
        { $$ = {}; }
    | ARRAY_START_TOKEN INTEGER_PATTERN ARRAY_END_TOKEN
        { $$ = $2; }
    ;

optionalInitializer
    : /* empty */
        { $$ = {}; }
    | EQUALS_TOKEN literal
        { driver.setDeclarationAssignment(); $$ = std::vector<silly::Literal>{ $2 }; }
    | LEFT_CURLY_BRACKET_TOKEN RIGHT_CURLY_BRACKET_TOKEN
        { driver.hasDeclarationHasInitializer(); $$ = {}; }
    | LEFT_CURLY_BRACKET_TOKEN initializerList RIGHT_CURLY_BRACKET_TOKEN
        { driver.hasDeclarationHasInitializer(); $$ = std::move( $2 ); }
    ;

initializerList
    : literal
        { $$ = std::vector<silly::Literal>{ $1 }; }
    | initializerList COMMA_TOKEN literal
        { $1.push_back( $3 ); $$ = std::move( $1 ); }
    ;

printStatement
    : PRINT_TOKEN printArgList optionalContinue
        { driver.enterPrintStatement( $2, @1 ); }
    ;

errorStatement
    : ERROR_TOKEN printArgList optionalContinue
        { driver.setPrintError(); driver.enterPrintStatement( $2, @1 ); }
    ;

printArgList
    : expression
        { $$ = std::vector<silly::Expr>{ $1 }; }
    | printArgList COMMA_TOKEN expression
        { $1.push_back( $3 ); $$ = std::move( $1 ); }
    ;

expression
    : expression BOOLEANOR_TOKEN expression
        { $$ = silly::Expr::makeBinaryOp( ExprOp::Or,  $1, $3 ); }
    | expression BOOLEANXOR_TOKEN expression
        { $$ = silly::Expr::makeBinaryOp( ExprOp::Xor, $1, $3 ); }
    | expression BOOLEANAND_TOKEN expression
        { $$ = silly::Expr::makeBinaryOp( ExprOp::And, $1, $3 ); }
    | expression EQUALITY_TOKEN expression
        { $$ = silly::Expr::makeBinaryOp( ExprOp::Equal, $1, $3 ); }
    | expression NOTEQUAL_TOKEN expression
        { $$ = silly::Expr::makeBinaryOp( ExprOp::NotEqual, $1, $3 ); }
    | expression LESSTHAN_TOKEN expression
        { $$ = silly::Expr::makeBinaryOp( ExprOp::Less, $1, $3 ); }
    | expression LESSEQUAL_TOKEN expression
        { $$ = silly::Expr::makeBinaryOp( ExprOp::LessEqual, $1, $3 ); }
    | expression GREATERTHAN_TOKEN expression
        { $$ = silly::Expr::makeBinaryOp( ExprOp::Greater, $1, $3 ); }
    | expression GREATEREQUAL_TOKEN expression
        { $$ = silly::Expr::makeBinaryOp( ExprOp::GreaterEqual, $1, $3 ); }
    | expression PLUSCHAR_TOKEN expression
        { $$ = silly::Expr::makeBinaryOp( ExprOp::Plus, $1, $3 ); }
    | expression MINUS_TOKEN expression
        { $$ = silly::Expr::makeBinaryOp( ExprOp::Minus, $1, $3 ); }
    | expression TIMES_TOKEN expression
        { $$ = silly::Expr::makeBinaryOp( ExprOp::Mul, $1, $3 ); }
    | expression DIV_TOKEN expression
        { $$ = silly::Expr::makeBinaryOp( ExprOp::Div, $1, $3 ); }
    | expression MOD_TOKEN expression
        { $$ = silly::Expr::makeBinaryOp( ExprOp::Mod, $1, $3 ); }
    | MINUS_TOKEN expression %prec UMINUS
        { $$ = silly::Expr::makeUnaryOp( ExprOp::Minus, $2 ); }
    | PLUSCHAR_TOKEN expression %prec UPLUS
        { $$ = silly::Expr::makeUnaryOp( ExprOp::Plus, $2 ); }
    | NOT_TOKEN expression %prec UNOT
        { $$ = silly::Expr::makeUnaryOp( ExprOp::Not, $2 ); }
    | BRACE_START_TOKEN expression BRACE_END_TOKEN
        { $$ = $2; }
    | literal
        { $$ = silly::Expr::fromLiteral( $1 ); }
    | IDENTIFIER
        { $$ = silly::Expr::fromVariable( $1 ); }
    | IDENTIFIER ARRAY_START_TOKEN expression ARRAY_END_TOKEN
        { $$ = silly::Expr::fromArrayVariable( $1, $3 ); }
    | CALL_TOKEN IDENTIFIER BRACE_START_TOKEN optionalCallArgList BRACE_END_TOKEN
        { $$ = silly::Expr::makeCall( $2 ); /* stub */ }
    ;

optionalContinue
    : /* empty */
    | CONTINUE_TOKEN
        { driver.setPrintContinue(); }
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
