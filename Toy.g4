grammar Toy;

startRule : statement* EOF;

statement
   : (declare | assignment | print)+
   ;

declare
   : DCL VARIABLENAME SEMICOLON
   ;

print
   : PRINT VARIABLENAME SEMICOLON
   ;

assignment
   : VARIABLENAME EQUALS rhs SEMICOLON
   ;

rhs
   : (element opertype element) | element
   ;

opertype
   : (MINUSCHAR | PLUSCHAR | TIMESCHAR | DIVCHAR)
   ;

element
   : (INTEGERLITERAL | VARIABLENAME)
   ;

EQUALS         : '=';
SEMICOLON      : ';';
MINUSCHAR      : '-';
TIMESCHAR      : '*';
DIVCHAR        : '/';
PLUSCHAR       : '+';
DCL            : 'DCL';
PRINT          : 'PRINT';
INTEGERLITERAL : (PLUSCHAR | MINUSCHAR)? [0-9]+;
VARIABLENAME   : [a-zA-Z]+;

WS : [ \t\r\n]+ -> skip;  // Skip spaces, tabs, newlines
