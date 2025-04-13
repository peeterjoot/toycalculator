grammar calculator;

STARTRULE : STATEMENT* EOF;

STATEMENT
   : (DECLARE | ASSIGNMENT)+
   ;

DECLARE
   : DCL VARIABLENAME SEMICOLON
   ;

ASSIGNMENT
   : VARIABLENAME EQUALS RHS SEMICOLON
   ;

RHS
   : ELEMENT OPERATOR ELEMENT
   ;

OPERATOR
   : (MINUSCHAR | PLUSCHAR | TIMESCHAR | DIVCHAR)
   ;

ELEMENT
   :
   (INTEGERLITERAL | VARIABLENAME)
   ;

EQUALS : '=';

SEMICOLON : ';';   

MINUSCHAR : '-';   

TIMESCHAR : '*';   

DIVCHAR : '/';   

PLUSCHAR : '+';   

INTEGERLITERAL : (PLUSCHAR | MINUSCHAR)? [0-9]+;               
                                                       
VARIABLENAME : [a-zA-Z]+;

DCL : 'DCL';
