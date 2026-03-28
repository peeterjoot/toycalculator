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
    #include <memory>
    #include <string>
    #include <vector>

    #include "location.hh"

    namespace silly
    {
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

        /// A string, location pair (identifier, array-size, ...)
        struct StringAndLoc
        {
            std::string name{};
            silly::location loc{};
        };

        struct TypeAndLoc
        {
            Types ty{};
            silly::location loc{};
        };

        struct TypeAndName
        {
            TypeAndLoc  t{};
            StringAndLoc id{};
        };

        struct Expr
        {
            enum class Kind
            {
                None,
                Int,
                Float,
                String,
                Bool,
                Variable,
                ArrayVariable,
                UnaryOp,
                BinaryOp,
                Call,
            };

            Kind kind{ Kind::None };
            silly::location loc{};
            std::string sval{}; /* int or float as string, or string content, or variable name */
            bool bval{};
            ExprOp op{ ExprOp::None };
            std::shared_ptr<Expr> left{};                /* UnaryOp: operand; BinaryOp: lhs; ArrayVariable: index */
            std::shared_ptr<Expr> right{};               /* BinaryOp: rhs */
            std::vector<std::shared_ptr<Expr>> params{}; /* Call: argument list */

            static Expr makeNone()
            {
                return { Kind::None };
            }

            static Expr makeInt( const std::string& s, const silly::location& loc )
            {
                return {
                    Kind::Int, loc,
                    s    // sval
                };
            }

            static Expr makeFloat( const std::string& s, const silly::location& loc )
            {
                return {
                    Kind::Float, loc,
                    s    // sval
                };
            }

            static Expr makeString( const std::string& s, const silly::location& loc )
            {
                return {
                    Kind::String, loc,
                    s    // sval
                };
            }

            static Expr makeBool( bool b, const silly::location& loc )
            {
                return {
                    Kind::Bool,
                    loc,
                    {},    // sval
                    b      // bval
                };
            }

            static Expr fromVariable( const silly::StringAndLoc& id )
            {
                return {
                    Kind::Variable, id.loc,
                    id.name        // sval
                };
            }

            static Expr fromArrayVariable( const silly::StringAndLoc& id, const Expr& idx )
            {
                Expr e{
                    Kind::ArrayVariable, id.loc,
                    id.name        // sval
                };
                e.left = std::make_shared<Expr>( idx );
                return e;
            }

            static Expr makeUnaryOp( ExprOp op, const Expr& operand, const silly::location& loc )
            {
                Expr e{ Kind::UnaryOp, loc };
                e.op = op;
                e.left = std::make_shared<Expr>( operand );
                return e;
            }

            static Expr makeBinaryOp( ExprOp op, const Expr& l, const Expr& r, const silly::location& loc )
            {
                Expr e{ Kind::BinaryOp, loc };
                e.op = op;
                e.left = std::make_shared<Expr>( l );
                e.right = std::make_shared<Expr>( r );
                return e;
            }

            static Expr makeCall( const silly::StringAndLoc& id, const std::vector<Expr>& args )
            {
                Expr e{
                    Kind::Call,
                    id.loc,
                    id.name     // sval
                };
                for ( const auto& a : args )
                    e.params.push_back( std::make_shared<Expr>( a ) );
                return e;
            }
        };
    }    // namespace silly
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

%code {
    #pragma clang diagnostic ignored "-Wunused-but-set-variable"
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
%type <silly::Expr>                         literal
%type <silly::StringAndLoc>                 identifier
%type <std::string>                         integerLiteral
%type <std::string>                         floatLiteral
%type <std::string>                         stringLiteral
%type <silly::TypeAndLoc>                   boolType
%type <silly::TypeAndLoc>                   intType
%type <silly::TypeAndLoc>                   floatType
%type <silly::TypeAndLoc>                   stringType
%type <silly::StringAndLoc>                 arrayBoundsExpression
%type <std::vector<silly::Expr>>            oneInitializerAsList
%type <std::vector<silly::Expr>>            initializerList
%type <std::string>                         importStatement
%type <silly::TypeAndLoc>                   scalarType
%type <std::vector<silly::Expr>>            printArgList
%type <silly::Expr>                         assignmentLHS
%type <silly::Expr>                         expression
%type <silly::TypeAndLoc>                   optionalReturnType
%type <std::vector<silly::TypeAndName>>     paramList
%type <std::vector<silly::TypeAndName>>     optionalParamList
%type <silly::TypeAndName>                  variableTypeAndName
%type <std::vector<silly::Expr>>            callArgList
%type <std::vector<silly::Expr>>            optionalCallArgList

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
        { driver.enterExitStatement( @1, $2 ); }
    | EXIT_TOKEN
        { driver.enterExitStatement( @1, silly::Expr::makeNone() ); }
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
    : CALL_TOKEN identifier BRACE_START_TOKEN optionalCallArgList BRACE_END_TOKEN
        { driver.enterCallStatement( @1, $2, $4 ); }
    ;

optionalCallArgList
    : /* empty */
        { $$ = {}; }
    | callArgList
        { $$ = std::move( $1 ); }
    ;

callArgList
    : expression
        { $$ = std::vector<silly::Expr>{ $1 }; }
    | callArgList COMMA_TOKEN expression
        { $1.push_back( $3 ); $$ = std::move( $1 ); }
    ;

getStatement
    : GET_TOKEN identifier
        { driver.enterGetStatement( @2, $2, silly::Expr::makeNone() ); }
    | GET_TOKEN identifier ARRAY_START_TOKEN expression ARRAY_END_TOKEN
        { driver.enterGetStatement( @2, $2, $4 ); }
    ;

forStatement
    : FOR_TOKEN
      BRACE_START_TOKEN intType identifier COLON_TOKEN
          BRACE_START_TOKEN
              expression COMMA_TOKEN expression
          BRACE_END_TOKEN
      BRACE_END_TOKEN
        { driver.enterForStatement( @1, $3, $4, $7, $9, silly::Expr::makeNone( ) ); }
      otherScopedStatements
        { driver.exitForStatement( @1 ); }
    | FOR_TOKEN
      BRACE_START_TOKEN intType identifier COLON_TOKEN
          BRACE_START_TOKEN
              expression COMMA_TOKEN expression COMMA_TOKEN expression
          BRACE_END_TOKEN
      BRACE_END_TOKEN
        { driver.enterForStatement( @1, $3, $4, $7, $9, $11 ); }
      otherScopedStatements
        { driver.exitForStatement( @1 ); }
    ;

ifElifElseStatement
    : ifStatement elifStatementList optionalElseStatement
        { driver.exitIfElifElseStatement( @1 ); }
    ;

ifStatement
    : IF_TOKEN BRACE_START_TOKEN expression BRACE_END_TOKEN
        { driver.enterIfStatement( @1, $3 ); }
      ifScopeStatements
    ;

elifStatementList
    : /* empty */
    | elifStatementList elifStatement
    ;

elifStatement
    : ELIF_TOKEN BRACE_START_TOKEN expression BRACE_END_TOKEN
        { driver.enterElifStatement( @1, $3 ); }
      ifScopeStatements
    ;

optionalElseStatement
    : /* empty */
    | ELSE_TOKEN
        { driver.enterElseStatement( @1 ); }
      ifScopeStatements
    ;

ifScopeStatements
    : LEFT_CURLY_BRACKET_TOKEN
        { driver.enterScopedStatements( @1 ); }
      statementList optionalReturnStatement RIGHT_CURLY_BRACKET_TOKEN
        { driver.exitScopedStatements( ); }
    ;

/* Antlr4ParseListener::enterScopedStatements: !isFunctionBody and !isForBody */
otherScopedStatements
    : LEFT_CURLY_BRACKET_TOKEN
      statementList optionalReturnStatement RIGHT_CURLY_BRACKET_TOKEN
    ;

optionalReturnStatement
    : /* empty */
    | RETURN_TOKEN expression ENDOFSTATEMENT_TOKEN
        { driver.enterReturnStatement( @1, $2 ); }
    | RETURN_TOKEN ENDOFSTATEMENT_TOKEN
        { driver.enterReturnStatement( @1, silly::Expr::makeNone( ) ); }
    ;

assignmentStatement
    : assignmentLHS EQUALS_TOKEN expression
        { driver.enterAssignmentStatement( $1, $3, @1, @3 ); }
    ;

assignmentLHS
    : identifier
        { $$ = silly::Expr::fromVariable( $1 ); }
    | identifier ARRAY_START_TOKEN expression ARRAY_END_TOKEN
        { $$ = silly::Expr::fromArrayVariable( $1, $3 ); }
    ;

importStatement
    : IMPORT_TOKEN identifier
        { driver.enterImportStatement( @1, $2 ); }
    ;

functionStatement
    : FUNCTION_TOKEN identifier
      BRACE_START_TOKEN optionalParamList BRACE_END_TOKEN
      optionalReturnType
        { driver.enterFunctionPrototype( @1, $2, $4, $6 ); }
    | FUNCTION_TOKEN identifier
      BRACE_START_TOKEN optionalParamList BRACE_END_TOKEN
      optionalReturnType
        { driver.enterFunctionDefinition( @1, $2, $4, $6 ); }
      otherScopedStatements
        { driver.exitFunctionDefinition( ); }
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
    : scalarType identifier
        { $$ = { $1, $2 }; }
    ;

optionalReturnType
    : /* empty */
        { $$ = {silly::Types::None}; }
    | COLON_TOKEN scalarType
        { $$ = $2; }
    ;

declareStatement
    : scalarType identifier arrayBoundsExpression
        { driver.enterDeclareStatement( $1, $2, $3 ); }
    | scalarType identifier arrayBoundsExpression EQUALS_TOKEN oneInitializerAsList
        { driver.enterDeclareStatement( $1, $2, $3, $5 ); }
    | scalarType identifier arrayBoundsExpression LEFT_CURLY_BRACKET_TOKEN initializerList RIGHT_CURLY_BRACKET_TOKEN
        { driver.enterDeclareStatement( $1, $2, $3, $5 ); }
    | scalarType identifier arrayBoundsExpression LEFT_CURLY_BRACKET_TOKEN RIGHT_CURLY_BRACKET_TOKEN
        { driver.enterDeclareStatementEmptyInit( $1, $2, $3 ); }
    | stringType identifier arrayBoundsExpression
        { driver.enterDeclareStatement( $1, $2, $3 ); }
    | stringType identifier arrayBoundsExpression EQUALS_TOKEN stringLiteral
        { driver.enterStringDeclareStatement( $1, $2, $3, $5 ); }
    | stringType identifier arrayBoundsExpression LEFT_CURLY_BRACKET_TOKEN stringLiteral RIGHT_CURLY_BRACKET_TOKEN
        { driver.enterStringDeclareStatement( $1, $2, $3, $5 ); }
    | stringType identifier arrayBoundsExpression LEFT_CURLY_BRACKET_TOKEN RIGHT_CURLY_BRACKET_TOKEN
        { driver.enterDeclareStatementEmptyInit( $1, $2, $3 ); }
    ;

scalarType
    : intType       { $$ = $1; }
    | floatType     { $$ = $1; }
    | boolType      { $$ = $1; }
    ;

intType
    : INT8_TOKEN    { $$ = {silly::Types::Int8, @1};  }
    | INT16_TOKEN   { $$ = {silly::Types::Int16, @1}; }
    | INT32_TOKEN   { $$ = {silly::Types::Int32, @1}; }
    | INT64_TOKEN   { $$ = {silly::Types::Int64, @1}; }
    ;

stringType
    : STRING_TOKEN  { $$ = {silly::Types::Int8, @1};  }

boolType
    : BOOL_TOKEN    { $$ = {silly::Types::Boolean, @1};  }
    ;

floatType
    : FLOAT32_TOKEN { $$ = {silly::Types::Float32, @1}; }
    | FLOAT64_TOKEN { $$ = {silly::Types::Float64, @1}; }
    ;

/* Returns -1 if no array bounds, otherwise the array size */
arrayBoundsExpression
    : /* empty */
        { $$ = {}; }
    | ARRAY_START_TOKEN INTEGER_PATTERN ARRAY_END_TOKEN
        { $$ = {$2, @2}; }
    ;

initializerList
    : expression
        { $$ = std::vector<silly::Expr>{ $1 }; }
    | initializerList COMMA_TOKEN expression
        { $1.push_back( $3 ); $$ = std::move( $1 ); }
    ;

oneInitializerAsList
    : expression
        { $$ = std::vector<silly::Expr>{ $1 }; }
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
        { $$ = silly::Expr::makeBinaryOp( ExprOp::Or,           $1, $3, @$ ); }
    | expression BOOLEANXOR_TOKEN expression
        { $$ = silly::Expr::makeBinaryOp( ExprOp::Xor,          $1, $3, @$ ); }
    | expression BOOLEANAND_TOKEN expression
        { $$ = silly::Expr::makeBinaryOp( ExprOp::And,          $1, $3, @$ ); }
    | expression EQUALITY_TOKEN expression
        { $$ = silly::Expr::makeBinaryOp( ExprOp::Equal,        $1, $3, @$ ); }
    | expression NOTEQUAL_TOKEN expression
        { $$ = silly::Expr::makeBinaryOp( ExprOp::NotEqual,     $1, $3, @$ ); }
    | expression LESSTHAN_TOKEN expression
        { $$ = silly::Expr::makeBinaryOp( ExprOp::Less,         $1, $3, @$ ); }
    | expression LESSEQUAL_TOKEN expression
        { $$ = silly::Expr::makeBinaryOp( ExprOp::LessEqual,    $1, $3, @$ ); }
    | expression GREATERTHAN_TOKEN expression
        { $$ = silly::Expr::makeBinaryOp( ExprOp::Greater,      $1, $3, @$ ); }
    | expression GREATEREQUAL_TOKEN expression
        { $$ = silly::Expr::makeBinaryOp( ExprOp::GreaterEqual, $1, $3, @$ ); }
    | expression PLUSCHAR_TOKEN expression
        { $$ = silly::Expr::makeBinaryOp( ExprOp::Plus,         $1, $3, @$ ); }
    | expression MINUS_TOKEN expression
        { $$ = silly::Expr::makeBinaryOp( ExprOp::Minus,        $1, $3, @$ ); }
    | expression TIMES_TOKEN expression
        { $$ = silly::Expr::makeBinaryOp( ExprOp::Mul,          $1, $3, @$ ); }
    | expression DIV_TOKEN expression
        { $$ = silly::Expr::makeBinaryOp( ExprOp::Div,          $1, $3, @$ ); }
    | expression MOD_TOKEN expression
        { $$ = silly::Expr::makeBinaryOp( ExprOp::Mod,          $1, $3, @$ ); }
    | MINUS_TOKEN expression %prec UMINUS
        { $$ = silly::Expr::makeUnaryOp( ExprOp::Minus, $2, @$ ); }
    | PLUSCHAR_TOKEN expression %prec UPLUS
        { $$ = silly::Expr::makeUnaryOp( ExprOp::Plus,  $2, @$ ); }
    | NOT_TOKEN expression %prec UNOT
        { $$ = silly::Expr::makeUnaryOp( ExprOp::Not,   $2, @$ ); }
    | BRACE_START_TOKEN expression BRACE_END_TOKEN
        { $$ = $2; }
    | literal
        { $$ = $1; }
    | identifier
        { $$ = silly::Expr::fromVariable( $1 ); }
    | identifier ARRAY_START_TOKEN expression ARRAY_END_TOKEN
        { $$ = silly::Expr::fromArrayVariable( $1, $3 ); }
    | CALL_TOKEN identifier BRACE_START_TOKEN optionalCallArgList BRACE_END_TOKEN
        { $$ = silly::Expr::makeCall( $2, $4 ); }
    ;

identifier
    : IDENTIFIER
        { $$ = { $1, @1 }; }
    ;

optionalContinue
    : /* empty */
    | CONTINUE_TOKEN
        { driver.setPrintContinue(); }
    ;

literal
    : integerLiteral
        { $$ = silly::Expr::makeInt( $1, @1 ); }
    | floatLiteral
        { $$ = silly::Expr::makeFloat( $1, @1 ); }
    | stringLiteral
        { $$ = silly::Expr::makeString( $1, @1 ); }
    | TRUE_LITERAL
        { $$ = silly::Expr::makeBool( true, @1 ); }
    | FALSE_LITERAL
        { $$ = silly::Expr::makeBool( false, @1 ); }
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
