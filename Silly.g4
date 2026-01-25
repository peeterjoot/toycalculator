//
// @file    Silly.g4
// @author  Peeter Joot <peeterjoot@pm.me>
// @brief   This is the antlr4 grammar for the silly compiler and language.
//
// @description
//
// This grammar implements a silly language that has:
// - a couple simple numeric operators (unary negation, binary +-*/),
// - an exit operation,
// - a declare operation, and typed declare operations (INT8, INT16, INT32, INT64, FLOAT32, FLOAT64, BOOL), plus array variants
// - an assignment operation,
// - a print operation, and
// - a get operation.
grammar Silly;

// Parser Rules (start with lower case)
// ====================================
// Entry point for the grammar, matching zero or more statements followed by optional EXIT, and then EOF.
startRule
  : (statement|comment)* (exitStatement ENDOFSTATEMENT_TOKEN)? comment* EOF
  ;

// A statement can be a declaration, assignment, PRINT, GET, ERROR, ABORT, IF, FOR, CALL, FUNCTION or comment.
statement
  : (callStatement | functionStatement | ifElifElseStatement |
     declareStatement | boolDeclareStatement | intDeclareStatement | floatDeclareStatement | stringDeclareStatement |
     assignmentStatement | printStatement | errorStatement | abortStatement | getStatement | forStatement
    )
    ENDOFSTATEMENT_TOKEN
  ;

ifElifElseStatement
  : ifStatement
    elifStatement*
    elseStatement?
  ;

ifStatement
  : IF_TOKEN BRACE_START_TOKEN expression BRACE_END_TOKEN LEFT_CURLY_BRACKET_TOKEN statement* RIGHT_CURLY_BRACKET_TOKEN
  ;

elifStatement
  : ELIF_TOKEN BRACE_START_TOKEN expression BRACE_END_TOKEN LEFT_CURLY_BRACKET_TOKEN statement* RIGHT_CURLY_BRACKET_TOKEN
  ;

elseStatement
  : ELSE_TOKEN LEFT_CURLY_BRACKET_TOKEN statement* RIGHT_CURLY_BRACKET_TOKEN
  ;

// For now both return and parameters, can only be scalar types.
functionStatement
  : FUNCTION_TOKEN IDENTIFIER
    BRACE_START_TOKEN (variableTypeAndName (COMMA_TOKEN variableTypeAndName)*)? BRACE_END_TOKEN
    (COLON_TOKEN scalarType)?
    LEFT_CURLY_BRACKET_TOKEN statement* returnStatement ENDOFSTATEMENT_TOKEN
    RIGHT_CURLY_BRACKET_TOKEN
  ;

// A declaration of a new variable (e.g., 'DCL x;' or 'DECLARE x;').  These are currently implicitly double.
declareStatement
  : (DCL_TOKEN|DECLARE_TOKEN)
    IDENTIFIER (arrayBoundsExpression)?
    ((EQUALS_TOKEN declareAssignmentExpression) |
     (LEFT_CURLY_BRACKET_TOKEN (expression (COMMA_TOKEN expression)*)? RIGHT_CURLY_BRACKET_TOKEN))?
  ;

boolDeclareStatement
  : BOOL_TOKEN
    IDENTIFIER (arrayBoundsExpression)?
    ((EQUALS_TOKEN declareAssignmentExpression) |
     (LEFT_CURLY_BRACKET_TOKEN (expression (COMMA_TOKEN expression)*)? RIGHT_CURLY_BRACKET_TOKEN))?
  ;

variableTypeAndName
  : scalarType IDENTIFIER
  ;

intDeclareStatement
  : (INT8_TOKEN | INT16_TOKEN | INT32_TOKEN | INT64_TOKEN)
    IDENTIFIER (arrayBoundsExpression)?
    ((EQUALS_TOKEN declareAssignmentExpression) |
     (LEFT_CURLY_BRACKET_TOKEN (expression (COMMA_TOKEN expression)*)? RIGHT_CURLY_BRACKET_TOKEN))?
  ;

floatDeclareStatement
  : (FLOAT32_TOKEN | FLOAT64_TOKEN)
    IDENTIFIER (arrayBoundsExpression)?
    ((EQUALS_TOKEN declareAssignmentExpression) |
     (LEFT_CURLY_BRACKET_TOKEN (expression (COMMA_TOKEN expression)*)? RIGHT_CURLY_BRACKET_TOKEN))?
  ;

// This is so that we can distinguish initialization-list expressions from assignment expressions.
// initialization-list expressions must be evaluatable at the point of declaration, so they can be expressions
// based on constants or parameters.  See the README for some examples.
declareAssignmentExpression
  : expression
  ;

stringDeclareStatement
  : STRING_TOKEN IDENTIFIER arrayBoundsExpression
    ((EQUALS_TOKEN STRING_PATTERN) | (LEFT_CURLY_BRACKET_TOKEN STRING_PATTERN RIGHT_CURLY_BRACKET_TOKEN))?
  ;

// A printStatement statement that outputs a list of variables, literals, or expressions (e.g., 'PRINT x, y, z;'), followed by a (optional) newline.
printStatement
  : PRINT_TOKEN expression (COMMA_TOKEN expression)* CONTINUE_TOKEN?
  ;

errorStatement
  : ERROR_TOKEN expression (COMMA_TOKEN expression)* CONTINUE_TOKEN?
  ;

// GET statement inputs into a scalar variable (e.g., 'GET x;', 'GET x[1]').
getStatement
  : GET_TOKEN scalarOrArrayElement
  ;

// An assignment of an expression to a variable (e.g., 'x = 42;').
assignmentStatement
  : scalarOrArrayElement EQUALS_TOKEN expression
  ;

// FOR ( INT32 x : (1, 11) ) { PRINT x; };
// FOR ( INT32 x : (1, 11, 2) ) { PRINT x; };
//
// respectively equivalent to:
//
// for ( int x = 1 ; x <= 10 ; x += 1 ) { ... }
// for ( int x = 1 ; x <= 10 ; x += 2 ) { ... }
forStatement
  : FOR_TOKEN
    BRACE_START_TOKEN (INT8_TOKEN | INT16_TOKEN | INT32_TOKEN | INT64_TOKEN) IDENTIFIER COLON_TOKEN
        BRACE_START_TOKEN
            forStart COMMA_TOKEN forEnd (COMMA_TOKEN forStep)?
        BRACE_END_TOKEN
    BRACE_END_TOKEN
    LEFT_CURLY_BRACKET_TOKEN statement* RIGHT_CURLY_BRACKET_TOKEN
  ;

forStart
  : forRangeExpression
  ;

forEnd
  : forRangeExpression
  ;

forStep
  : forRangeExpression
  ;

forRangeExpression
  : expression
  ;

// Implicit or explicit exit from a program (e.g., 'EXIT;' ('EXIT 0;'), 'EXIT 3;', 'EXIT x;')
exitStatement
  : EXIT_TOKEN expression?
  ;

returnStatement
  : RETURN_TOKEN expression?
  ;

callStatement
  : callExpression
  ;

callExpression
  : CALL_TOKEN IDENTIFIER parameterList
  ;

parameterList
  : BRACE_START_TOKEN (parameterExpression (COMMA_TOKEN parameterExpression)*)? BRACE_END_TOKEN
  ;

parameterExpression
  : expression
  ;

scalarOrArrayElement
  : IDENTIFIER (indexExpression)?
  ;

indexExpression
  : ARRAY_START_TOKEN expression ARRAY_END_TOKEN
  ;

// ─────────────────────────────────────────────────────────────
//   New expression hierarchy

expression
  : binaryExpressionLowest                      # exprLowest
  ;

binaryExpressionLowest
  : binaryExpressionOr (BOOLEANOR_TOKEN binaryExpressionOr)*     # orExpr
  ;

binaryExpressionOr
  : binaryExpressionXor (BOOLEANXOR_TOKEN binaryExpressionXor)*  # xorExpr
  ;

binaryExpressionXor
  : binaryExpressionAnd (BOOLEANAND_TOKEN binaryExpressionAnd)*  # andExpr
  ;

binaryExpressionAnd
  : binaryExpressionCompare (equalityOperator binaryExpressionCompare )?   # eqNeExpr
  ;

binaryExpressionCompare
  : binaryExpressionAddSub (relationalOperator binaryExpressionAddSub )?   # compareExpr
  ;

binaryExpressionAddSub
  : binaryExpressionMulDiv (additionOperator binaryExpressionMulDiv )*      # addSubExpr
  ;

binaryExpressionMulDiv
  : unaryExpression ( multiplicativeOperator unaryExpression )*                  # mulDivExpr
  ;

unaryExpression
  : unaryOperator unaryExpression                                 # unaryOp
  | primaryExpression                                             # primary
  ;

primaryExpression
  : literal                                                       # litPrimary
  | scalarOrArrayElement                                          # varPrimary
  | callExpression                                                # callPrimary
  | BRACE_START_TOKEN expression BRACE_END_TOKEN                  # parenExpr
  ;

// Operator rules to yield a clean vector of operator nodes (avoids separate vectors per token type)
equalityOperator
  : EQUALITY_TOKEN | NOTEQUAL_TOKEN
  ;

relationalOperator
  : LESSTHAN_TOKEN | LESSEQUAL_TOKEN | GREATERTHAN_TOKEN | GREATEREQUAL_TOKEN
  ;

additionOperator
  : PLUSCHAR_TOKEN | MINUS_TOKEN
  ;

multiplicativeOperator
  : TIMES_TOKEN | DIV_TOKEN
  ;
// ─────────────────────────────────────────────────────────────

// A single-line comment
comment
  : COMMENT_SKIP_RULE
  ;

abortStatement
  : ABORT_TOKEN
  ;

scalarType
  : INT8_TOKEN | INT16_TOKEN | INT32_TOKEN | INT64_TOKEN | FLOAT32_TOKEN | FLOAT64_TOKEN | BOOL_TOKEN
  ;

arrayBoundsExpression
  : ARRAY_START_TOKEN INTEGER_PATTERN ARRAY_END_TOKEN
  ;

// A binary operator for addition, subtraction, multiplication, or division, ...
binaryOperator
  : MINUS_TOKEN | PLUSCHAR_TOKEN | TIMES_TOKEN | DIV_TOKEN
  | LESSTHAN_TOKEN | GREATERTHAN_TOKEN | LESSEQUAL_TOKEN | GREATEREQUAL_TOKEN
  | EQUALITY_TOKEN | NOTEQUAL_TOKEN
  | BOOLEANOR_TOKEN | BOOLEANAND_TOKEN | BOOLEANXOR_TOKEN
  ;

predicateOperator
  : LESSTHAN_TOKEN | GREATERTHAN_TOKEN | LESSEQUAL_TOKEN | GREATEREQUAL_TOKEN
  | EQUALITY_TOKEN | NOTEQUAL_TOKEN
  ;

// An optional unary operator for positive or negative (e.g., '+' or '-').
unaryOperator
  : MINUS_TOKEN | PLUSCHAR_TOKEN | NOT_TOKEN
  ;

booleanLiteral
  : BOOLEAN_PATTERN
  ;

integerLiteral
  : INTEGER_PATTERN
  ;

numericLiteral
  : INTEGER_PATTERN | FLOAT_PATTERN
  ;

literal
  : INTEGER_PATTERN | FLOAT_PATTERN | BOOLEAN_PATTERN | STRING_PATTERN
  ;

/////////////////////////////////////////////////////////////////////////////////
//
// LEXER Rules
// ===========

// Matches unsigned integer literals, (e.g., '42', '123', '7').  signed literals produced with unaryExpression
INTEGER_PATTERN
  : [0-9]+
  ;

BOOLEAN_PATTERN
  : TRUE_LITERAL | FALSE_LITERAL
  ;

STRING_PATTERN
  : DQUOTE_TOKEN (~["])* DQUOTE_TOKEN
  ;
// Could allow for escaped quotes, but let's get the simple case working first:
//  : DQUOTE_TOKEN (~["\\] | '\\' .)* DQUOTE_TOKEN

// Matches unsigned floating point literals.  Examples:
// 123
// 123.3334
// 123.3334E0
// 42
// 42.3334
// 42.3334E7
// 7
// 7.3334
// 7.3334E-1
FLOAT_PATTERN
  : [0-9]+( DECIMALSEP_TOKEN [0-9]+)? (EXPONENT_TOKEN MINUS_TOKEN? [0-9]+)?
  ;

// Matches single-line comments (e.g., '// comment') and skips them.
COMMENT_SKIP_RULE
  : '//' ~[\r\n]* -> skip
  ;

// Matches the equals sign for assignments (e.g., '=').
EQUALS_TOKEN
  : '='
  ;

DQUOTE_TOKEN
  : '"'
  ;

// Matches the equality operator (e.g., 'EQ').
EQUALITY_TOKEN
  : 'EQ'
  ;

// Matches the equality operator (e.g., 'NE').
NOTEQUAL_TOKEN
  : 'NE'
  ;

// Matches the semicolon that terminates statements (e.g., ';').
ENDOFSTATEMENT_TOKEN
  : ';'
  ;

// Matches the minus sign for subtraction or negation (e.g., '-').
MINUS_TOKEN
  : '-'
  ;

// Matches the multiplication operator (e.g., '*').
TIMES_TOKEN
  : '*'
  ;

// Matches a North american decimal separator
DECIMALSEP_TOKEN
  : '.'
  ;

// Matches a North american decimal separator
EXPONENT_TOKEN
  : 'E'
  ;

// Matches the division operator (e.g., '/').
DIV_TOKEN
  : '/'
  ;

COMMA_TOKEN
  : ','
  ;

ARRAY_START_TOKEN
  : '['
  ;

ARRAY_END_TOKEN
  : ']'
  ;

BRACE_START_TOKEN
  : '('
  ;

BRACE_END_TOKEN
  : ')'
  ;

LEFT_CURLY_BRACKET_TOKEN
  : '{'
  ;

RIGHT_CURLY_BRACKET_TOKEN
  : '}'
  ;

COLON_TOKEN
  : ':'
  ;

// Matches the plus sign for addition or positive (e.g., '+').
PLUSCHAR_TOKEN
  : '+'
  ;

LESSTHAN_TOKEN
  : '<'
  ;

GREATERTHAN_TOKEN
  : '>'
  ;

LESSEQUAL_TOKEN
  : '<='
  ;

GREATEREQUAL_TOKEN
  : '>='
  ;

// Matches the 'OR' keyword for boolean or bitwise OR
BOOLEANOR_TOKEN
  : 'OR'
  ;

// Matches the 'AND' keyword for boolean or bitwise AND
BOOLEANAND_TOKEN
  : 'AND'
  ;

// Matches the 'XOR' keyword for boolean or bitwise XOR
BOOLEANXOR_TOKEN
  : 'XOR'
  ;

// Matches the 'NOT' keyword for BOOL inversion
NOT_TOKEN
  : 'NOT'
  ;

// Matches the 'DCL' keyword for declarations.
DCL_TOKEN
  : 'DCL'
  ;

// signed integers of various sizes, and ieee float/double types:
INT8_TOKEN
  : 'INT8'
  ;

INT16_TOKEN
  : 'INT16'
  ;

INT32_TOKEN
  : 'INT32'
  ;

INT64_TOKEN
  : 'INT64'
  ;

FLOAT32_TOKEN
  : 'FLOAT32'
  ;

FLOAT64_TOKEN
  : 'FLOAT64'
  ;

// Boolean tokens:
BOOL_TOKEN
  : 'BOOL'
  ;

FOR_TOKEN
  : 'FOR'
  ;

IF_TOKEN
  : 'IF'
  ;

ELSE_TOKEN
  : 'ELSE'
  ;

ELIF_TOKEN
  : 'ELIF'
  ;

TRUE_LITERAL
  : 'TRUE'
  ;

FALSE_LITERAL
  : 'FALSE'
  ;

// Matches the 'DECLARE' keyword for declarations.
DECLARE_TOKEN
  : 'DECLARE'
  ;

// Matches the 'PRINT' keyword
PRINT_TOKEN
  : 'PRINT'
  ;

CONTINUE_TOKEN
  : 'CONTINUE'
  ;

// Matches the 'ERROR' keyword
ERROR_TOKEN
  : 'ERROR'
  ;

// Matches the 'ABORT' keyword
ABORT_TOKEN
  : 'ABORT'
  ;

// Matches the 'GET' keyword
GET_TOKEN
  : 'GET'
  ;

// Matches the 'EXIT' keyword
EXIT_TOKEN
  : 'EXIT'
  ;

RETURN_TOKEN
  : 'RETURN'
  ;

STRING_TOKEN
  : 'STRING'
  ;

FUNCTION_TOKEN
  : 'FUNCTION'
  ;

CALL_TOKEN
  : 'CALL'
  ;

// Matches variable names (e.g., 'x', 'foo', 'my_var'), consisting of letters (any case), numbers and underscores, but starting with a letter.
// This rule must be after the TOKENs above to prohibit variables like 'INT32 INT32;' (error_keyword_declare.silly)
IDENTIFIER
  : [a-zA-Z][a-zA-Z0-9_]*
  ;

// Matches whitespace (spaces, tabs, newlines) and skips it.
WS
  : [ \t\r\n]+ -> skip
  ;
