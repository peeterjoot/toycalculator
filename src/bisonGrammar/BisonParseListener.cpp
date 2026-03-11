///
/// @file    BisonParseListener.cpp
/// @author  Peeter Joot <peeterjoot@pm.me>
/// @brief   Bison based experimental parse tree listener and MLIR builder.
///
#include <mlir/Dialect/Arith/IR/Arith.h>

#include "BisonParseListener.hpp"
#include "DriverState.hpp"
#include "LocationStack.hpp"
#include "PrintFlags.hpp"
#include "SillyDialect.hpp"
#include "silly.lex.hh"    // flex-generated reentrant scanner
#include "silly.tab.hh"

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

    BisonParseListener::BisonParseListener( silly::SourceManager& s, const std::string& filename )
        : Builder{ s, filename }
    {
    }

    yyscan_t BisonParseListener::getScanner()
    {
        return scanner;
    }

    mlir::Location BisonParseListener::getLocation( const silly::BisonParser::location_type& bLoc )
    {
        mlir::FileLineColLoc startLoc =
            mlir::FileLineColLoc::get( builder.getStringAttr( sourceFile ), bLoc.begin.line, bLoc.begin.column );

        return startLoc;
    }

    LocPairs BisonParseListener::getLocations( const silly::BisonParser::location_type& bLoc )
    {
        mlir::FileLineColLoc startLoc =
            mlir::FileLineColLoc::get( builder.getStringAttr( sourceFile ), bLoc.begin.line, bLoc.begin.column );
        mlir::FileLineColLoc endLoc =
            mlir::FileLineColLoc::get( builder.getStringAttr( sourceFile ), bLoc.end.line, bLoc.end.column );

        return { startLoc, endLoc };
    }

    void BisonParseListener::enterStartRule( const silly::BisonParser::location_type& bLoc )
    {
        mlir::Location loc = getLocation( bLoc );

        createMain( loc, loc );
    }

    void BisonParseListener::exitStartRule( const silly::BisonParser::location_type& bLoc )
    {
        assert( !isModule );    // TODO

        LocPairs locs = getLocations( bLoc );

        llvm::SmallVector<mlir::Location, 2> funcLocs{ locs.first, locs.second };
        mlir::Location fLoc = builder.getFusedLoc( funcLocs );

        ParserPerFunctionState& f = funcState( currentFuncName );
        mlir::func::FuncOp funcOp = f.getFuncOp();
        funcOp->setLoc( fLoc );

        // HACK: unconditional:
        createMainExit( locs.second );
    }

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

    void BisonParseListener::enterPrintStatement( Literal lit, const silly::BisonParser::location_type& bLoc,
                                        const silly::BisonParser::location_type& valueLoc )
    {
        // printf( "%s:%d:%d:PRINT %d (value at %d:%d)\n", sourceFile.c_str(), bLoc.begin.line, bLoc.begin.column,
        //         value, valueLoc.begin.line, valueLoc.begin.column );
        mlir::Location loc = getLocation( bLoc );
        PrintFlags pf = PRINT_FLAGS_NONE;
        LocationStack ls( builder, loc );
        std::vector<mlir::Value> vargs;
        // for ( SillyParser::ExpressionContext *parg : args )
        {
            // mlir::Value v = parseExpression( parg, {}, ls );
            mlir::Location vLoc = getLocation( valueLoc );
            mlir::Value v;
            switch ( lit.kind )
            {
                case Literal::Kind::Int:
                    v = parseInteger( vLoc, 64, lit.sval, ls );
                    break;
                case Literal::Kind::Float:
                    v = parseFloat( vLoc, typ.f64, lit.sval, ls );
                    break;
                case Literal::Kind::Bool:
                    //v = parseInteger( vLoc, 1, lit.sval, ls );
                    v = mlir::arith::ConstantIntOp::create( builder, vLoc, lit.bval, 1 );
                    break;
                case Literal::Kind::String:
                    v = buildStringLiteral( vLoc, lit.sval, ls );
                    break;
            }

            if ( !v )
            {
                emitInternalError( loc, __FILE__, __LINE__, __func__, "parseExpression failed", currentFuncName );
                return;
            }

            vargs.push_back( v );
        }

        // ls.push_back( loc );
        mlir::arith::ConstantIntOp constFlagOp = mlir::arith::ConstantIntOp::create( builder, loc, pf, 32 );

        silly::PrintOp::create( builder, loc, constFlagOp, vargs );
    }

    void BisonParseListener::emitParseError( const silly::BisonParser::location_type& bLoc, const std::string& msg )
    {
        mlir::Location loc = getLocation( bLoc );
        emitInternalError( loc, __FILE__, __LINE__, __func__, msg, currentFuncName );
    }

    mlir::OwningOpRef<mlir::ModuleOp> runParseListener( silly::SourceManager& s, const std::string& filename )
    {
        BisonParseListener listener( s, filename );

        return listener.run();
    }
}    // namespace silly

// vim: et ts=4 sw=4
