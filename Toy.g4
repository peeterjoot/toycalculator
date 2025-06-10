//
// @file    Toy.g4
// @author  Peeter Joot <peeterjoot@pm.me>
// @brief   This is the antlr4 grammar for the Toy compiler project.
//
// @description
//
// This grammar implements a toy (calculator) language that has:
// - a couple simple numeric operators (unary negation, binary +-*/),
// - an exit operation,
// - a declare operation, and typed declare operations (INT8, INT16, INT32, INT64, FLOAT32, FLOAT64, BOOL)
// - an assignment operation, and
// - a print operation.
grammar Toy;

// Parser Rules (start with lower case)
// ====================================
// Entry point for the grammar, matching zero or more statements followed by optional EXIT, and then EOF.
startRule
  : (statement|comment)* (exitStatement ENDOFSTATEMENT_TOKEN)? comment* EOF
  ;

// A statement can be a declaration, assignment, print, or comment.
statement
  : (ifelifelse | declare | boolDeclare | intDeclare | floatDeclare | stringDeclare | assignment | print) ENDOFSTATEMENT_TOKEN
  ;

ifelifelse
  : IF_TOKEN BRACE_START_TOKEN booleanValue BRACE_END_TOKEN SCOPE_START_TOKEN statement* SCOPE_END_TOKEN
    (ELIF_TOKEN BRACE_START_TOKEN booleanValue BRACE_END_TOKEN SCOPE_START_TOKEN statement* SCOPE_END_TOKEN)*
    (ELSE_TOKEN SCOPE_START_TOKEN statement* SCOPE_END_TOKEN)?
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
  : (DCL_TOKEN|DECLARE_TOKEN) VARIABLENAME_PATTERN (arrayBoundsExpression)?
  ;

boolDeclare
  : BOOL_TOKEN VARIABLENAME_PATTERN (arrayBoundsExpression)?
  ;

intDeclare
  : (INT8_TOKEN | INT16_TOKEN | INT32_TOKEN | INT64_TOKEN) VARIABLENAME_PATTERN (arrayBoundsExpression)?
  ;

floatDeclare
  : (FLOAT32_TOKEN | FLOAT64_TOKEN) VARIABLENAME_PATTERN (arrayBoundsExpression)?
  ;

stringDeclare
  : STRING_TOKEN VARIABLENAME_PATTERN arrayBoundsExpression
  ;

arrayBoundsExpression
  : ARRAY_START_TOKEN INTEGER_PATTERN ARRAY_END_TOKEN
  ;

// Implicit or explicit exit from a program (e.g., 'EXIT;' ('EXIT 0;'), 'EXIT 3;', 'EXIT x;')
exitStatement
  : EXIT_TOKEN (numericLiteral | VARIABLENAME_PATTERN)?
  ;

// A print statement that outputs a variable (e.g., 'PRINT x;').
print
  : PRINT_TOKEN (VARIABLENAME_PATTERN | STRING_PATTERN)
  ;

// An assignment of an expression to a variable (e.g., 'x = 42;').
//assignment
//  : VARIABLENAME_PATTERN (INDEX_EXPRESSION)? EQUALS_TOKEN assignmentExpression
//  ;
assignment
  : VARIABLENAME_PATTERN EQUALS_TOKEN assignmentExpression
  ;

// The right-hand side of an assignment, either a binary or unary expression.
assignmentExpression
  : literal
  | unaryOperator? VARIABLENAME_PATTERN
  | binaryElement binaryOperator binaryElement
  ;

binaryElement
  : numericLiteral
  | unaryOperator? VARIABLENAME_PATTERN
  ;

booleanElement
  : booleanLiteral | VARIABLENAME_PATTERN
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

//INDEX_EXPRESSION
//  : ARRAY_START_TOKEN (VARIABLENAME_PATTERN | INTEGER_PATTERN) ARRAY_END_TOKEN
//  ;

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

// Matches the 'EXIT' keyword for print statements.
EXIT_TOKEN
  : 'EXIT'
  ;

STRING_TOKEN
  : 'STRING'
  ;

// Matches variable names (e.g., 'x', 'foo'), consisting of letters (any case) and numbers, but starting with a letter.
VARIABLENAME_PATTERN
  : [a-zA-Z][a-zA-Z0-9]*
  ;

// Matches whitespace (spaces, tabs, newlines) and skips it.
WS
  : [ \t\r\n]+ -> skip
  ;
