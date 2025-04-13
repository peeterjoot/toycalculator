grammar calculator;

startrule : statement* EOF;

statement
   : (declare | assignment)+
   ;

declare
   : DCL VARIABLENAME SEMICOLON
   ;

assignment
   : VARIABLENAME EQUALS rhs SEMICOLON
   ;

rhs
   : element operator element
   ;

operator
   : (MINUSCHAR | PLUSCHAR | TIMESCHAR | DIVCHAR)
   ;

element
   : (INTEGERLITERAL | VARIABLENAME)
   ;

EQUALS : '=';
SEMICOLON : ';';   
MINUSCHAR : '-';   
TIMESCHAR : '*';   
DIVCHAR : '/';   
PLUSCHAR : '+';   
DCL : 'DCL';

INTEGERLITERAL : (PLUSCHAR | MINUSCHAR)? [0-9]+;               
                                                       
VARIABLENAME : [a-zA-Z]+;
