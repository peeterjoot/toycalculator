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

// Parser Rules
// ============
// Entry point for the grammar, matching zero or more statements followed by optional EXIT, and then EOF.
startRule
  : (statement|comment)* (exitStatement ENDOFSTATEMENT)? comment* EOF
  ;

// A statement can be a declaration, assignment, print, or comment.
statement
  : (declare | boolDeclare | intDeclare | floatDeclare | assignment | print) ENDOFSTATEMENT
  ;

// A single-line comment, handled by the COMMENT lexer token.
comment
  : COMMENT
  ;

// A declaration of a new variable (e.g., 'DCL x;' or 'DECLARE x;').  These are currently implicitly double.
declare
  : (DCL|DECLARE) VARIABLENAME
  ;

boolDeclare
  : BOOL VARIABLENAME
  ;

intDeclare
  : (INT8|INT16|INT32|INT64) VARIABLENAME
  ;

floatDeclare
  : (FLOAT32|FLOAT64) VARIABLENAME
  ;

// Implicit or explicit exit from a program (e.g., 'EXIT;' ('EXIT 0;'), 'EXIT 3;', 'EXIT x;')
exitStatement
  : EXIT (numericLiteral | VARIABLENAME)?
  ;

// A print statement that outputs a variable (e.g., 'PRINT x;').
print
  : PRINT VARIABLENAME
  ;

// An assignment of an expression to a variable (e.g., 'x = 42;').
assignment
  : VARIABLENAME EQUALS assignmentExpression
  ;

// The right-hand side of an assignment, either a binary or unary expression.
assignmentExpression
  : literal
  | unaryOperator? VARIABLENAME
  | binaryElement binaryOperator binaryElement
  ;

binaryElement
  : numericLiteral
  | unaryOperator? VARIABLENAME
  ;

// A binary operator for addition, subtraction, multiplication, or division.
binaryOperator
  : MINUSCHAR | PLUSCHAR | TIMESCHAR | DIVCHAR | LESSTHAN | GREATERTHAN | LESSEQUAL | GREATEREQUAL | EQUAL | BOOLEANOR | BOOLEANAND | BOOLEANXOR
  ;

// An optional unary operator for positive or negative (e.g., '+' or '-').
unaryOperator
  : MINUSCHAR | PLUSCHAR
  ;

numericLiteral
  : INTEGERLITERAL | FLOATLITERAL
  ;

literal
  : INTEGERLITERAL | FLOATLITERAL | BOOLEANLITERAL
  ;

// Lexer Rules
// ===========
// Matches integer literals, optionally signed (e.g., '42', '-123', '+7').
INTEGERLITERAL
  : (PLUSCHAR | MINUSCHAR)? [0-9]+
  ;

BOOLEANLITERAL
  : TRUE | FALSE
  ;

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
FLOATLITERAL
  : (PLUSCHAR | MINUSCHAR)? [0-9]+( DECIMALSEP [0-9]+)? (EXPONENT MINUSCHAR? [0-9]+)?
  ;

// Matches single-line comments (e.g., '// comment') and skips them.
COMMENT
  : '//' ~[\r\n]* -> skip
  ;

// Matches the equals sign for assignments (e.g., '=').
EQUALS
  : '='
  ;

// Matches the semicolon that terminates statements (e.g., ';').
ENDOFSTATEMENT
  : ';'
  ;

// Matches the minus sign for subtraction or negation (e.g., '-').
MINUSCHAR
  : '-'
  ;

// Matches the multiplication operator (e.g., '*').
TIMESCHAR
  : '*'
  ;

// Matches a North american decimal separator
DECIMALSEP
  : '.'
  ;

// Matches a North american decimal separator
EXPONENT
  : 'E'
  ;

// Matches the division operator (e.g., '/').
DIVCHAR
  : '/'
  ;

// Matches the plus sign for addition or positive (e.g., '+').
PLUSCHAR
  : '+'
  ;

LESSTHAN
  : '<'
  ;

GREATERTHAN
  : '>'
  ;

LESSEQUAL
  : '<='
  ;

GREATEREQUAL
  : '>='
  ;

EQUAL
  : '=='
  ;

BOOLEANOR
  : 'OR'
  ;

BOOLEANAND
  : 'AND'
  ;

BOOLEANXOR
  : 'XOR'
  ;

// Matches the 'DCL' keyword for declarations.
DCL
  : 'DCL'
  ;

// signed integers of various sizes, and ieee float/double types:
INT8 : 'INT8' ;
INT16 : 'INT16' ;
INT32 : 'INT32' ;
INT64 : 'INT64' ;
FLOAT32 : 'FLOAT32' ;
FLOAT64 : 'FLOAT64' ;

// Boolean tokens:
BOOL : 'BOOL' ;
TRUE : 'TRUE' ;
FALSE : 'FALSE' ;

// Matches the 'DECLARE' keyword for declarations.
DECLARE
  : 'DECLARE'
  ;

// Matches the 'PRINT' keyword for print statements.
PRINT
  : 'PRINT'
  ;

// Matches the 'EXIT' keyword for print statements.
EXIT
  : 'EXIT'
  ;

// Matches variable names (e.g., 'x', 'foo'), consisting of letters (any case) and numbers, but starting with a letter.
VARIABLENAME
  : [a-zA-Z][a-zA-Z0-9]*
  ;

// Matches whitespace (spaces, tabs, newlines) and skips it.
WS
  : [ \t\r\n]+ -> skip
  ;
