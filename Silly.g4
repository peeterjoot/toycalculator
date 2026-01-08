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

// A statement can be a declaration, assignment, print, get, if, for, comment.
statement
  : (call | function | ifelifelse | declare | boolDeclare | intDeclare | floatDeclare | stringDeclare | assignment | print | get | for) ENDOFSTATEMENT_TOKEN
  ;

ifelifelse
  : ifStatement
    elifStatement*
    elseStatement?
  ;

ifStatement
  : IF_TOKEN BRACE_START_TOKEN booleanValue BRACE_END_TOKEN SCOPE_START_TOKEN statement* SCOPE_END_TOKEN
  ;

elifStatement
  : ELIF_TOKEN BRACE_START_TOKEN booleanValue BRACE_END_TOKEN SCOPE_START_TOKEN statement* SCOPE_END_TOKEN
  ;

elseStatement
  : ELSE_TOKEN SCOPE_START_TOKEN statement* SCOPE_END_TOKEN
  ;

// For now both return and parameters, can only be scalar types.
function
  : FUNCTION_TOKEN IDENTIFIER BRACE_START_TOKEN (variableTypeAndName (COMMA_TOKEN variableTypeAndName)*)? BRACE_END_TOKEN (COLON_TOKEN scalarType)? SCOPE_START_TOKEN statement* returnStatement ENDOFSTATEMENT_TOKEN SCOPE_END_TOKEN
  ;

booleanValue
  : booleanElement | (binaryElement predicateOperator binaryElement)
  ;

// A single-line comment
comment
  : COMMENT_SKIP_RULE
  ;

// A declaration of a new variable (e.g., 'DCL x;' or 'DECLARE x;').  These are currently implicitly double.
declare
  : (DCL_TOKEN|DECLARE_TOKEN) IDENTIFIER (arrayBoundsExpression)?
  ;

boolDeclare
  : BOOL_TOKEN IDENTIFIER (arrayBoundsExpression)?
  ;

variableTypeAndName
  //: IDENTIFIER COLON_TOKEN scalarType
  : scalarType IDENTIFIER
  ;

scalarType
  : INT8_TOKEN | INT16_TOKEN | INT32_TOKEN | INT64_TOKEN | FLOAT32_TOKEN | FLOAT64_TOKEN | BOOL_TOKEN
  ;

intDeclare
  : (INT8_TOKEN | INT16_TOKEN | INT32_TOKEN | INT64_TOKEN) IDENTIFIER (arrayBoundsExpression)?
  ;

floatDeclare
  : (FLOAT32_TOKEN | FLOAT64_TOKEN) IDENTIFIER (arrayBoundsExpression)?
  ;

stringDeclare
  : STRING_TOKEN IDENTIFIER arrayBoundsExpression
  ;

arrayBoundsExpression
  : ARRAY_START_TOKEN INTEGER_PATTERN ARRAY_END_TOKEN
  ;

// Implicit or explicit exit from a program (e.g., 'EXIT;' ('EXIT 0;'), 'EXIT 3;', 'EXIT x;')
exitStatement
  : EXIT_TOKEN (numericLiteral | scalarOrArrayElement)?
  ;

returnStatement
  : RETURN_TOKEN (literal | scalarOrArrayElement)?
  ;

// A print statement that outputs a list of variables (e.g., 'PRINT x, y, z;'), followed by a newline.
print
  : PRINT_TOKEN printArgument (COMMA_TOKEN printArgument)*
  ;

printArgument
  : scalarOrArrayElement | STRING_PATTERN | numericLiteral | booleanLiteral
  ;

// A get statement that inputs into a scalar variable (e.g., 'GET x;', 'GET x[1]').
get
  : GET_TOKEN scalarOrArrayElement
  ;

// An assignment of an expression to a variable (e.g., 'x = 42;').
assignment
  : scalarOrArrayElement EQUALS_TOKEN assignmentRvalue
  ;

assignmentRvalue
  : rvalueExpression
  ;

// FOR ( x : (1, 11) ) { PRINT x; };
// FOR ( x : (1, 11, 2) ) { PRINT x; };
//
// respectively equivalent to:
//
// for ( x = 1 ; x <= 10 ; x += 1 ) { ... }
// for ( x = 1 ; x <= 10 ; x += 2 ) { ... }
for
  : FOR_TOKEN
    BRACE_START_TOKEN IDENTIFIER COLON_TOKEN
        BRACE_START_TOKEN
            forStart COMMA_TOKEN forEnd (COMMA_TOKEN forStep)?
        BRACE_END_TOKEN
    BRACE_END_TOKEN
    SCOPE_START_TOKEN statement* SCOPE_END_TOKEN
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
  : rvalueExpression
  ;

// The right-hand side of an assignment or a parameter, either a binary or unary expression.
rvalueExpression
  : literal
  | unaryOperator? scalarOrArrayElement
  | binaryElement binaryOperator binaryElement
  | call
  ;

call
  : CALL_TOKEN IDENTIFIER parameterList
  ;

parameterList
  : BRACE_START_TOKEN (parameterExpression (COMMA_TOKEN parameterExpression)*)? BRACE_END_TOKEN
  ;

parameterExpression
  : rvalueExpression
  ;

binaryElement
  : numericLiteral
  | unaryOperator? scalarOrArrayElement
  ;

booleanElement
  : booleanLiteral | scalarOrArrayElement
  ;

scalarOrArrayElement
  : IDENTIFIER (indexExpression)?
  ;

indexExpression
// probably want (allow: Now t[i+1] or t[someFunc()], ..., to parse correctly.)
// ARRAY_START_TOKEN assignmentExpression ARRAY_END_TOKEN
  : ARRAY_START_TOKEN (IDENTIFIER | INTEGER_PATTERN) ARRAY_END_TOKEN
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

numericLiteral
  : INTEGER_PATTERN | FLOAT_PATTERN
  ;

literal
  : INTEGER_PATTERN | FLOAT_PATTERN | BOOLEAN_PATTERN | STRING_PATTERN
  ;

booleanLiteral
  : INTEGER_PATTERN | BOOLEAN_PATTERN
  ;

// Lexer Rules
// ===========

// Matches integer literals, optionally signed (e.g., '42', '-123', '+7').
INTEGER_PATTERN
  : (PLUSCHAR_TOKEN | MINUS_TOKEN)? [0-9]+
  ;

BOOLEAN_PATTERN
  : TRUE_LITERAL | FALSE_LITERAL
  ;

STRING_PATTERN
  : DQUOTE_TOKEN (~["])* DQUOTE_TOKEN
  ;
// Could allow for escaped quotes, but let's get the simple case working first:
//  : DQUOTE_TOKEN (~["\\] | '\\' .)* DQUOTE_TOKEN

// Matches floating point literal.  Examples:
// 42
// -123
// +7
// 42.3334
// -123.3334
// +7.3334
// 42.3334E7
// -123.3334E0
// +7.3334E-1
FLOAT_PATTERN
  : (PLUSCHAR_TOKEN | MINUS_TOKEN)? [0-9]+( DECIMALSEP_TOKEN [0-9]+)? (EXPONENT_TOKEN MINUS_TOKEN? [0-9]+)?
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

SCOPE_START_TOKEN
  : '{'
  ;

SCOPE_END_TOKEN
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

// Matches the 'PRINT' keyword for print statements.
PRINT_TOKEN
  : 'PRINT'
  ;

// Matches the 'GET' keyword for get statements.
GET_TOKEN
  : 'GET'
  ;

// Matches the 'EXIT' keyword for print statements.
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
