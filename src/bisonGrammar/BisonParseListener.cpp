///
/// @file    BisonParseListener.cpp
/// @author  Peeter Joot <peeterjoot@pm.me>
/// @brief   Bison based experimental parse tree listener and MLIR builder.
///
#include <mlir/Dialect/Arith/IR/Arith.h>

#include <format>

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

    void BisonParseListener::setModule()
    {
        isModule = true;
    }

    void BisonParseListener::setPrintContinue()
    {
        hasPrintContinue = true;
    }

    void BisonParseListener::setPrintError()
    {
        hasPrintError = true;
    }

    void BisonParseListener::setDeclarationAssignment()
    {
        declarationAssignmentInitialization = true;
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
        if ( !isModule )
        {
            mlir::Location loc = getLocation( bLoc );

            createMain( loc, loc );
        }
    }

    void BisonParseListener::exitStartRule( const silly::BisonParser::location_type& bLoc )
    {
        if ( !isModule )
        {
            LocPairs locs = getLocations( bLoc );

            llvm::SmallVector<mlir::Location, 2> funcLocs{ locs.first, locs.second };
            mlir::Location fLoc = builder.getFusedLoc( funcLocs );

            ParserPerFunctionState& f = funcState( currentFuncName );
            mlir::func::FuncOp funcOp = f.getFuncOp();
            funcOp->setLoc( fLoc );

            // HACK: unconditional:
            createMainExit( locs.second );
        }
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

    // eventually: parseExpression
    mlir::Value BisonParseListener::parsePrintArg( mlir::Location vLoc, const silly::LiteralOrVariable& parg, LocationStack &ls )
    {
        mlir::Value v;
        if ( parg.kind == LiteralOrVariable::Kind::Literal )
        {
            switch ( parg.lit.kind )
            {
                case Literal::Kind::None:
                    break;
                case Literal::Kind::Int:
                    v = parseInteger( vLoc, 64, parg.lit.sval, ls );
                    break;
                case Literal::Kind::Float:
                    v = parseFloat( vLoc, typ.f64, parg.lit.sval, ls );
                    break;
                case Literal::Kind::Bool:
                    v = mlir::arith::ConstantIntOp::create( builder, vLoc, parg.lit.bval, 1 );
                    break;
                case Literal::Kind::String:
                    v = buildStringLiteral( vLoc, parg.lit.sval, ls );
                    break;
            }
            if ( !v )
            {
                emitInternalError( vLoc, __FILE__, __LINE__, __func__,
                                   std::format( "parseExpression failed. Kind: {}", (int)parg.lit.kind ),
                                   currentFuncName );
                return v;
            }
        }
        else
        {
            v = variableToValue( vLoc, parg.name, {}, vLoc, ls );
        }

        return v;
    }

    void BisonParseListener::enterPrintStatement( const std::vector<silly::LiteralOrVariable>& args,
                                                  const silly::BisonParser::location_type& printLoc )
    {
        mlir::Location loc = getLocation( printLoc );
        PrintFlags pf = PRINT_FLAGS_NONE;
        if ( hasPrintContinue )
        {
            pf = PRINT_FLAGS_CONTINUE;
        }

        if ( hasPrintError )
        {
            pf = (PrintFlags)( pf | PRINT_FLAGS_ERROR );
        }

        hasPrintContinue = false;
        hasPrintError = false;

        LocationStack ls( builder, loc );
        std::vector<mlir::Value> vargs;
        for ( const silly::LiteralOrVariable& parg : args )
        {
            mlir::Location vLoc = loc;    // per-argument location?
            mlir::Value v = parsePrintArg( vLoc, parg, ls );
            if ( !v )
            {
                return;
            }

            vargs.push_back( v );
        }

        // ls.push_back( loc );
        mlir::arith::ConstantIntOp constFlagOp = mlir::arith::ConstantIntOp::create( builder, loc, pf, 32 );

        silly::PrintOp::create( builder, loc, constFlagOp, vargs );
    }

    mlir::Type BisonParseListener::integerDeclarationType( mlir::Location loc, const std::string& typeName )
    {
        mlir::Type ty{};

        if ( typeName == "INT8" )
        {
            ty = typ.i8;
        }
        else if ( typeName == "INT16" )
        {
            ty = typ.i16;
        }
        else if ( typeName == "INT32" )
        {
            ty = typ.i32;
        }
        else if ( typeName == "INT64" )
        {
            ty = typ.i64;
        }
        else
        {
            emitInternalError( loc, __FILE__, __LINE__, __func__, "Unsupported signed integer declaration size.",
                               currentFuncName );
        }

        return ty;
    }

    void BisonParseListener::declarationHelper( mlir::Location tLoc, mlir::Location aLoc, const std::string& varName,
                                                const std::string& arraySizeString, mlir::Type ty, Literal initializer,
                                                LocationStack& ls )
    {
        bool haveInitializer{};
        std::vector<mlir::Value> initializers;

        if ( initializer.kind != Literal::Kind::None )
        {
            haveInitializer = true;
            mlir::Value init;

            // proper location for initializer?
            if ( ty == typ.i1 )
            {
                init = mlir::arith::ConstantIntOp::create( builder, tLoc, initializer.bval, 1 );
            }
            else if ( mlir::IntegerType ity = mlir::dyn_cast<mlir::IntegerType>( ty ) )
            {
                init = parseInteger( tLoc, ity.getWidth(), initializer.sval, ls );
            }
            else
            {
                mlir::FloatType fty = mlir::dyn_cast<mlir::FloatType>( ty );

                init = parseFloat( tLoc, fty, initializer.sval, ls );
            }

            if ( !init )
            {
                emitInternalError( tLoc, __FILE__, __LINE__, __func__, "parseExpression failed", currentFuncName );
                return;
            }

            initializers.push_back( init );
        }

        registerDeclaration( tLoc, varName, ty, aLoc, arraySizeString, haveInitializer, initializers, ls );
    }

    void BisonParseListener::enterIntDeclare( const std::string& typeName, const std::string& varName,
                                              const std::string& arraySizeString, Literal initializer,
                                              const silly::BisonParser::location_type& typeLoc,
                                              const silly::BisonParser::location_type& nameLoc,
                                              const silly::BisonParser::location_type& arrayLoc )
    {
        mlir::Location tLoc = getLocation( typeLoc );
        // bool initIsDeclare = declarationAssignmentInitialization;
        // declarationAssignmentInitialization = false;
        mlir::Type ty = integerDeclarationType( tLoc, typeName );
        if ( !ty )
        {
            return;
        }
        mlir::Location aLoc = getLocation( arrayLoc );
        LocationStack ls( builder, tLoc );


        declarationHelper( tLoc, aLoc, varName, arraySizeString, ty, initializer, ls );
    }

    void BisonParseListener::enterFloatDeclare( const std::string& typeName, const std::string& varName,
                                                const std::string& arraySizeString, Literal initializer,
                                                const silly::BisonParser::location_type& typeLoc,
                                                const silly::BisonParser::location_type& nameLoc,
                                                const silly::BisonParser::location_type& arrayLoc )
    {
        mlir::Location tLoc = getLocation( typeLoc );
        // bool initIsDeclare = declarationAssignmentInitialization;
        // declarationAssignmentInitialization = false;
        mlir::Type ty;
        if ( typeName == "FLOAT32" )
        {
            ty = typ.f32;
        }
        else if ( typeName == "FLOAT64" )
        {
            ty = typ.f64;
        }
        else
        {
            emitInternalError( tLoc, __FILE__, __LINE__, __func__, "Unsupported floating point declaration size.",
                               currentFuncName );
            return;
        }

        mlir::Location aLoc = getLocation( arrayLoc );
        LocationStack ls( builder, tLoc );

        declarationHelper( tLoc, aLoc, varName, arraySizeString, ty, initializer, ls );
    }

    void BisonParseListener::enterBoolDeclare( const std::string& varName, const std::string& arraySizeString,
                                               Literal initializer, const silly::BisonParser::location_type& typeLoc,
                                               const silly::BisonParser::location_type& nameLoc,
                                               const silly::BisonParser::location_type& arrayLoc )
    {
        mlir::Location tLoc = getLocation( typeLoc );
        // bool initIsDeclare = declarationAssignmentInitialization;
        // declarationAssignmentInitialization = false;
        mlir::Type ty = typ.i1;
        mlir::Location aLoc = getLocation( arrayLoc );
        LocationStack ls( builder, tLoc );

        declarationHelper( tLoc, aLoc, varName, arraySizeString, ty, initializer, ls );
    }

    void BisonParseListener::enterAssignment( const std::string& varName, const silly::LiteralOrVariable& rhs,
                                              const silly::BisonParser::location_type& lhsLoc,
                                              const silly::BisonParser::location_type& rhsLoc )
    {
        mlir::Location loc = getLocation( lhsLoc );
        LocationStack ls( builder, loc );

        // SillyParser::IndexExpressionContext *indexExpr = lhs->indexExpression();
        mlir::Value currentIndexExpr = mlir::Value{};

        bool declared = isVariableDeclared( varName );
        if ( !declared )
        {
            // coverage: syntax-error/undeclared-var.silly
            emitUserError( loc, std::format( "Attempt to assign to undeclared variable: {}", varName ),
                           currentFuncName );
            return;
        }

#if 0
        if ( indexExpr )
        {
            currentIndexExpr = parseExpression( indexExpr->expression(), {}, ls );
            if ( !currentIndexExpr )
            {
                emitInternalError( loc, __FILE__, __LINE__, __func__, "parseExpression failed", currentFuncName );
                return;
            }
        }
#endif

        mlir::Location aLoc = getLocation( rhsLoc );
        mlir::Value resultValue = parsePrintArg( aLoc, rhs, ls );
        if ( !resultValue )
        {
            return;
        }

        processAssignment( aLoc, resultValue, varName, currentIndexExpr, ls );
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
