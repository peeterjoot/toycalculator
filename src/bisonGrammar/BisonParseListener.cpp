#include "BisonParseListener.hpp"
#include "silly_bison.tab.hh"

// Include the flex-generated header for the reentrant scanner
#include "silly_bison.lex.hh"

namespace silly
{
    bool BisonParseListener::parse()
    {
        FILE* f = fopen( filename.c_str(), "r" );
        if ( !f )
        {
            fprintf( stderr, "cannot open %s\n", filename.c_str() );
            return false;
        }

        yylex_init( &scanner );
        yyset_extra( this, scanner );
        yyset_in( f, scanner );

        silly::BisonParser parser( *this );
        int result = parser.parse();

        yylex_destroy( scanner );
        fclose( f );

        return result == 0 && errorCount == 0;
    }

    void BisonParseListener::emitPrint( int value, const silly::BisonParser::location_type& loc )
    {
        printf( "%s:%d:%d:PRINT %d\n", filename.c_str(), loc.begin.line, loc.begin.column, value );
        // Replace with your MLIR builder call
    }

    void BisonParseListener::emitError( const silly::BisonParser::location_type& loc, const std::string& msg )
    {
        fprintf( stderr, "%s:%d:%d: error: %s\n", filename.c_str(), loc.begin.line, loc.begin.column, msg.c_str() );
        errorCount++;
    }
}    // namespace silly

#if 1
int main( int argc, char** argv )
{
    silly::BisonParseListener dr( "foo.silly" );
    dr.parse();

    return 0;
}
#endif
