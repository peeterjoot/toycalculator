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

            if ( !hasExplicitExit )
            {
                createMainExit( locs.second );
            }
        }
    }

    void BisonParseListener::enterExitStatement( const silly::BisonParser::location_type& exitLoc,
                                                 const silly::Expr& var )
    {
        hasExplicitExit = true;

        mlir::Location loc = getLocation( exitLoc );
        LocationStack ls( builder, loc );

        mlir::Type returnType = findReturnType();
        mlir::Value value = parsePrintArg( loc, returnType, var, ls );
        processReturnLike( loc, value, ls );
    }

    void BisonParseListener::enterExitStatement( const silly::BisonParser::location_type& exitLoc )
    {
        hasExplicitExit = true;

        mlir::Location loc = getLocation( exitLoc );
        LocationStack ls( builder, loc );

        processReturnLike( loc, {}, ls );
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
    mlir::Value BisonParseListener::parsePrintArg( mlir::Location vLoc, mlir::Type ty,
                                                   const silly::Expr& parg, LocationStack& ls )
    {
        mlir::Value v;
        if ( parg.kind == Expr::Kind::Literal )
        {
            switch ( parg.lit.kind )
            {
                case Literal::Kind::None:
                    break;
                case Literal::Kind::Int:
                {
                    int width{ 64 };
                    if ( ty )
                    {
                        mlir::IntegerType ity = mlir::cast<mlir::IntegerType>( ty );
                        width = ity.getWidth();
                    }
                    v = parseInteger( vLoc, width, parg.lit.sval, ls );
                    break;
                }
                case Literal::Kind::Float:
                {
                    mlir::FloatType fty = typ.f64;
                    if ( ty )
                    {
                        fty = mlir::cast<mlir::FloatType>( ty );
                    }
                    v = parseFloat( vLoc, fty, parg.lit.sval, ls );
                    break;
                }
                case Literal::Kind::Bool:
                {
                    v = mlir::arith::ConstantIntOp::create( builder, vLoc, parg.lit.bval, 1 );
                    break;
                }
                case Literal::Kind::String:
                {
                    v = buildStringLiteral( vLoc, parg.lit.sval, ls );
                    break;
                }
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
            mlir::Value index;
            if ( parg.kind == Expr::Kind::ArrayVariable )
            {
                index = parseInteger( vLoc, 64, parg.index, ls );
            }

#if 0
            // FIXME: this is where things now go bad when I have expressions.

fedoravm:/home/peeter/toycalculator/src/bisonGrammar> gdb `which silly`
Reading symbols from /home/peeter/toycalculator/build/bin/silly...
Breakpoint 1 at 0x4304d0
Breakpoint 2 at 0x42fc10
Breakpoint 3 at 0x43391c: file /home/peeter/toycalculator/src/driver/Builder.cpp, line 382.
Downloading separate debug info for system-supplied DSO at 0xfffff7ffa000
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/lib64/libthread_db.so.1".

Breakpoint 3, silly::Builder::lookupDeclareForVar (this=0xffffffffdf88, loc=..., varName=...)
    at /home/peeter/toycalculator/src/driver/Builder.cpp:382
382             silly::DeclareOp declareOp{};
(gdb) n
383             ParserPerFunctionState &f = funcState( currentFuncName );
(gdb) p currentFuncName
$1 = "main"
(gdb) p varName
$2 = ""
(gdb) up
#1  0x0000000000433b00 in silly::Builder::variableToValue (this=0xffffffffdf88, loc=..., varName="", iValue=..., iLoc=..., ls=...)
    at /home/peeter/toycalculator/src/driver/Builder.cpp:416
416                 silly::DeclareOp declareOp = lookupDeclareForVar( loc, varName );
(gdb) up
#2  0x00000000004c7850 in silly::BisonParseListener::parsePrintArg (this=0xffffffffdf88, vLoc=..., ty=..., parg=..., ls=...)
    at /home/peeter/toycalculator/src/bisonGrammar/BisonParseListener.cpp:224
224                 v = variableToValue( vLoc, parg.name, index, vLoc, ls );
(gdb) p parg
$3 = (const silly::Expr &) @0x60d400: {kind = silly::Expr::Kind::BinaryOp, lit = {kind = silly::Literal::Kind::None, sval = "",
    bval = false}, name = "", op = "+", index = "", left = std::shared_ptr<silly::Expr> (use count 1, weak count 0) = {get() = 0x58b810},
  right = std::shared_ptr<silly::Expr> (use count 1, weak count 0) = {get() = 0x58ba00}}


(gdb) bt
#0  silly::Builder::lookupDeclareForVar (this=0xffffffffdf88, loc=..., varName="") at /home/peeter/toycalculator/src/driver/Builder.cpp:382
#1  0x0000000000433b00 in silly::Builder::variableToValue (this=0xffffffffdf88, loc=..., varName="", iValue=..., iLoc=..., ls=...)
    at /home/peeter/toycalculator/src/driver/Builder.cpp:416
#2  0x00000000004c7850 in silly::BisonParseListener::parsePrintArg (this=0xffffffffdf88, vLoc=..., ty=..., parg=..., ls=...)
    at /home/peeter/toycalculator/src/bisonGrammar/BisonParseListener.cpp:224
#3  0x00000000004c8a1c in silly::BisonParseListener::enterAssignmentStatement (this=0xffffffffdf88, var=..., rhs=..., lhsLoc=..., rhsLoc=...)
    at /home/peeter/toycalculator/src/bisonGrammar/BisonParseListener.cpp:435
#4  0x00000000004ca3d0 in silly::BisonParser::parse (this=0xffffffffdf00) at /home/peeter/toycalculator/src/bisonGrammar/silly.y:373
#5  0x00000000004c7a1c in silly::BisonParseListener::run (this=0xffffffffdf88)

#endif
            v = variableToValue( vLoc, parg.name, index, vLoc, ls );
        }

        return v;
    }

    void BisonParseListener::enterPrintStatement( const std::vector<silly::Expr>& args,
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
        for ( const silly::Expr& parg : args )
        {
            mlir::Location vLoc = loc;    // per-argument location?
            mlir::Value v = parsePrintArg( vLoc, {}, parg, ls );
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

    void BisonParseListener::enterIntDeclareStatement( const std::string& typeName, const std::string& varName,
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

    void BisonParseListener::enterFloatDeclareStatement( const std::string& typeName, const std::string& varName,
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

    void BisonParseListener::enterBoolDeclareStatement( const std::string& varName, const std::string& arraySizeString,
                                                        Literal initializer,
                                                        const silly::BisonParser::location_type& typeLoc,
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

    void BisonParseListener::enterAssignmentStatement( const silly::Expr& var,
                                                       const silly::Expr& rhs,
                                                       const silly::BisonParser::location_type& lhsLoc,
                                                       const silly::BisonParser::location_type& rhsLoc )
    {
        mlir::Location loc = getLocation( lhsLoc );
        LocationStack ls( builder, loc );

        bool declared = isVariableDeclared( var.name );
        if ( !declared )
        {
            // coverage: syntax-error/undeclared-var.silly
            emitUserError( loc, std::format( "Attempt to assign to undeclared variable: {}", var.name ),
                           currentFuncName );
            return;
        }

        mlir::Value indexValue = mlir::Value{};
        if ( var.kind == Expr::Kind::ArrayVariable )
        {
            mlir::Location rLoc = getLocation( rhsLoc );
            indexValue = parseInteger( rLoc, 64, var.index, ls );
            if ( !indexValue )
            {
                emitInternalError( loc, __FILE__, __LINE__, __func__, "parseExpression failed", currentFuncName );
                return;
            }
        }

        mlir::Location aLoc = getLocation( rhsLoc );
        mlir::Value resultValue = parsePrintArg( aLoc, {}, rhs, ls );
        if ( !resultValue )
        {
            return;
        }

        processAssignment( aLoc, resultValue, var.name, indexValue, ls );
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
