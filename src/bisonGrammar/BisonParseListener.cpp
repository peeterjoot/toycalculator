///
/// @file    BisonParseListener.cpp
/// @author  Peeter Joot <peeterjoot@pm.me>
/// @brief   Bison based experimental parse tree listener and MLIR builder.
///
#include "BisonParseListener.hpp"
#include "silly_bison.tab.hh"
#include "DriverState.hpp"

#include "silly_bison.lex.hh" // flex-generated reentrant scanner

namespace silly
{
    class FILEManager
    {
       public:
        FILE* f{};

        ~FILEManager()
        {
            if ( f )
            {
                fclose( f );
            }
        }
    };

    mlir::OwningOpRef<mlir::ModuleOp> BisonParseListener::run()
    {
        driverState.openFailed = false;

        FILEManager file;
        file.f = fopen( sourceFile.c_str(), "r" );
        if ( !file.f )
        {
            driverState.openFailed = true;

            return nullptr;
        }

        yylex_init( &scanner );
        yyset_extra( this, scanner );
        yyset_in( file.f, scanner );

        silly::BisonParser parser( *this );
        int result = parser.parse();

        yylex_destroy( scanner );

        if ( ( result == 0 ) and ( errorCount == 0 ) )
        {
            return std::move( rmod );
        }

        return nullptr;
    }

    void BisonParseListener::emitPrint( int value, const silly::BisonParser::location_type& printLoc,
                                        const silly::BisonParser::location_type& valueLoc )
    {
        printf( "%s:%d:%d:PRINT %d (value at %d:%d)\n", sourceFile.c_str(), printLoc.begin.line, printLoc.begin.column,
                value, valueLoc.begin.line, valueLoc.begin.column );
    }

    void BisonParseListener::emitError( const silly::BisonParser::location_type& loc, const std::string& msg )
    {
        fprintf( stderr, "%s:%d:%d: error: %s\n", sourceFile.c_str(), loc.begin.line, loc.begin.column, msg.c_str() );
        errorCount++;
    }

    mlir::OwningOpRef<mlir::ModuleOp> runParseListener( silly::SourceManager& s, const std::string& filename )
    {
        BisonParseListener listener( s, filename );

        return listener.run();
    }
}    // namespace silly

// vim: et ts=4 sw=4
