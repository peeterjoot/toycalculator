grammar Toy;

// Parser Rules
// ============
// Entry point for the grammar, matching zero or more statements followed by optional RETURN, and then EOF.
// FIXME: this should prohibit comments after RETURN.
startRule
  : (statement|comment)* (returnStatement ENDOFSTATEMENT)? EOF
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

// Implicit or explicit return from a program (e.g., 'RETURN;', 'RETURN 3;')  Return without value equivalent to 'RETURN 0;'
returnStatement
  : RETURN element*
  ;

// A print statement that outputs a variable (e.g., 'PRINT x;').
print
  : PRINT VARIABLENAME
  ;

// An assignment of an expression to a variable (e.g., 'x = 42;').
assignment
  : VARIABLENAME EQUALS rhs
  ;

// The right-hand side of an assignment, either a binary or unary expression.
rhs
  : binaryExpression
  | unaryExpression
  ;

// A binary expression with two elements and an operator (e.g., 'x + 42').
binaryExpression
  : element binaryOperator element
  ;

// A unary expression with an optional operator and an element (e.g., '-x').
unaryExpression
  : unaryOperator element
  ;

// A binary operator for addition, subtraction, multiplication, or division.
binaryOperator
  : (MINUSCHAR | PLUSCHAR | TIMESCHAR | DIVCHAR)
  ;

// An optional unary operator for positive or negative (e.g., '+' or '-').
unaryOperator
  : (MINUSCHAR | PLUSCHAR)?
  ;

// An element in an expression, either an integer literal or a variable name.
element
  : (INTEGERLITERAL | VARIABLENAME | FLOATLITERAL | TRUE | FALSE)
  ;

// Lexer Rules
// ===========
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

// Matches the 'RETURN' keyword for print statements.
RETURN
  : 'RETURN'
  ;

// Matches integer literals, optionally signed (e.g., '42', '-123', '+7').
INTEGERLITERAL
  : (PLUSCHAR | MINUSCHAR)? [0-9]+
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

// Matches variable names (e.g., 'x', 'foo'), consisting of letters (any case) and numbers, but starting with a letter.
VARIABLENAME
  : [a-zA-Z][a-zA-Z0-9]*
  ;

// Matches whitespace (spaces, tabs, newlines) and skips it.
WS
  : [ \t\r\n]+ -> skip
  ;
