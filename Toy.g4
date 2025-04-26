grammar Toy;

// Parser Rules
// ============
// Entry point for the grammar, matching zero or more statements followed by EOF.
startRule
  : (statement|comment)* EOF
  ;

// A statement can be a declaration, assignment, print, or comment.
statement
  : (declare | assignment | print) ENDOFSTATEMENT
  ;

// A single-line comment, handled by the COMMENT lexer token.
comment
  : COMMENT
  ;

// A declaration of a new variable (e.g., 'DCL x;' or 'DECLARE x;').
declare
  : (DCL|DECLARE) VARIABLENAME
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
  : binaryexpression
  | unaryexpression
  ;

// A binary expression with two elements and an operator (e.g., 'x + 42').
binaryexpression
  : element binaryoperator element
  ;

// A unary expression with an optional operator and an element (e.g., '-x').
unaryexpression
  : unaryoperator element
  ;

// A binary operator for addition, subtraction, multiplication, or division.
binaryoperator
  : (MINUSCHAR | PLUSCHAR | TIMESCHAR | DIVCHAR)
  ;

// An optional unary operator for positive or negative (e.g., '+' or '-').
unaryoperator
  : (MINUSCHAR | PLUSCHAR)?
  ;

// An element in an expression, either an integer literal or a variable name.
element
  : (INTEGERLITERAL | VARIABLENAME)
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

// Matches the 'DECLARE' keyword for declarations.
DECLARE
  : 'DECLARE'
  ;

// Matches the 'PRINT' keyword for print statements.
PRINT
  : 'PRINT'
  ;

// Matches integer literals, optionally signed (e.g., '42', '-123', '+7').
INTEGERLITERAL
  : (PLUSCHAR | MINUSCHAR)? [0-9]+
  ;

// Matches variable names (e.g., 'x', 'foo'), consisting of letters (any case) and numbers, but starting with a letter.
VARIABLENAME
  : [a-zA-Z][a-zA-Z0-9]*
  ;

// Matches whitespace (spaces, tabs, newlines) and skips it.
WS
  : [ \t\r\n]+ -> skip
  ;
