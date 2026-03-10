%option noyywrap nounput noinput
%option reentrant
%option bison-bridge
%option bison-locations
%option extra-type="silly::BisonParseListener*"

%{
#include "silly_bison.tab.hh"
#include <climits>
#include <cerrno>
#include "BisonParseListener.hpp"

#define YY_USER_ACTION                        \
    yylloc->begin = yylloc->end;              \
    for ( const char* c = yytext; *c; ++c ) { \
        if ( *c == '\n' ) {                   \
            yylloc->end.line++;               \
            yylloc->end.column = 1;           \
        } else {                              \
            yylloc->end.column++;             \
        }                                     \
    }
%}

%%

"PRINT"             { return silly::BisonParser::token::PRINT_TOKEN; }
";"                 { return silly::BisonParser::token::ENDOFSTATEMENT_TOKEN; }
[0-9]+  {
    errno = 0;
    char* end;
    long val = std::strtol( yytext, &end, 10 );

    if ( errno == ERANGE || val > INT_MAX || val < INT_MIN ) {
        silly::BisonParseListener* driver = yyget_extra( yyscanner );
        driver->emitError( *yylloc, "integer literal out of range" );
        yylval->emplace<int>( 0 );  // recover with a dummy value
    } else {
        yylval->emplace<int>( static_cast<int>( val ) );
    }
    return silly::BisonParser::token::INTEGER_PATTERN;
}
[ \t\r\n]+          { /* skip whitespace */ }
"//"[^\n]*          { /* skip comments */ }
.                   { return yytext[0]; }

%%
