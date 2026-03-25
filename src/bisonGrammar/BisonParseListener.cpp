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
#include "helper.hpp"
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

            ParserPerFunctionState& f = lookupFunctionState( currentFuncName );
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

        mlir::Type returnType = getReturnType();
        mlir::Value value = parseExpression( loc, returnType, var, ls );
        createReturn( loc, value, ls );
    }

    void BisonParseListener::enterExitStatement( const silly::BisonParser::location_type& exitLoc )
    {
        hasExplicitExit = true;

        mlir::Location loc = getLocation( exitLoc );
        LocationStack ls( builder, loc );

        createReturn( loc, {}, ls );
    }

    void BisonParseListener::enterAbortStatement( const silly::BisonParser::location_type& bLoc )
    {
        mlir::Location loc = getLocation( bLoc );

        silly::AbortOp::create( builder, loc );
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

        mlir::Location loc = builder.getUnknownLoc();
        // shouldn't see this if there was a non-internal error logged:
        emitInternalError( loc, __FILE__, __LINE__, __func__,
                           "Catastrophic compilation failure: Failed to generate MLIR module", currentFuncName );
        return nullptr;
    }

    mlir::Value BisonParseListener::parseExpression( mlir::Location vLoc, mlir::Type ty, const silly::Expr& parg,
                                                     LocationStack& ls )
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
                    v = createIntegerFromString( vLoc, width, parg.lit.sval, ls );
                    break;
                }
                case Literal::Kind::Float:
                {
                    mlir::FloatType fty = typ.f64;
                    if ( ty )
                    {
                        fty = mlir::cast<mlir::FloatType>( ty );
                    }
                    v = createFloatFromString( vLoc, fty, parg.lit.sval, ls );
                    break;
                }
                case Literal::Kind::Bool:
                {
                    v = mlir::arith::ConstantIntOp::create( builder, vLoc, parg.lit.bval, 1 );
                    break;
                }
                case Literal::Kind::String:
                {
                    v = createStringLiteral( vLoc, parg.lit.sval, ls );
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
        else if ( parg.kind == Expr::Kind::UnaryOp )
        {
            mlir::Value left = parseExpression( vLoc, ty, *parg.left, ls );
            UnaryOp uop = UnaryOp::Undefined;
            switch ( parg.op )
            {
                case ExprOp::Minus:
                {
                    uop = UnaryOp::Negate;
                    break;
                }
                case ExprOp::Plus:
                {
                    uop = UnaryOp::Plus;
                    break;
                }
                case ExprOp::Not:
                {
                    uop = UnaryOp::Not;
                    break;
                }
                default:
                {
                    emitInternalError( vLoc, __FILE__, __LINE__, __func__,
                                       std::format( "parseExpression failed. Unknown unary op: {}", (int)parg.kind ),
                                       currentFuncName );
                }
            }

            if ( uop != UnaryOp::Undefined )
            {
                v = createUnary( vLoc, left, uop, ls );
            }
        }
        else if ( parg.kind == Expr::Kind::BinaryOp )
        {
            mlir::Value left = parseExpression( vLoc, ty, *parg.left, ls );
            mlir::Value right = parseExpression( vLoc, ty, *parg.right, ls );
            mlir::Type bty = biggestTypeOf( left.getType(), right.getType() );
            switch ( parg.op )
            {
                case ExprOp::Mul:
                {
                    v = createBinaryArith( vLoc, silly::ArithBinOpKind::Mul, bty, left, right, ls );
                    break;
                }
                case ExprOp::Div:
                {
                    v = createBinaryArith( vLoc, silly::ArithBinOpKind::Div, bty, left, right, ls );
                    break;
                }
                case ExprOp::Mod:
                {
                    v = createBinaryArith( vLoc, silly::ArithBinOpKind::Mod, bty, left, right, ls );
                    break;
                }
                case ExprOp::Plus:
                {
                    v = createBinaryArith( vLoc, silly::ArithBinOpKind::Add, bty, left, right, ls );
                    break;
                }
                case ExprOp::Minus:
                {
                    v = createBinaryArith( vLoc, silly::ArithBinOpKind::Sub, bty, left, right, ls );
                    break;
                }
                case ExprOp::Or:
                {
                    v = createBinaryArith( vLoc, silly::ArithBinOpKind::Or, bty, left, right, ls );
                    break;
                }
                case ExprOp::And:
                {
                    v = createBinaryArith( vLoc, silly::ArithBinOpKind::And, bty, left, right, ls );
                    break;
                }
                case ExprOp::Xor:
                {
                    v = createBinaryArith( vLoc, silly::ArithBinOpKind::Xor, bty, left, right, ls );
                    break;
                }
                case ExprOp::Equal:
                {
                    v = createBinaryCompare( vLoc, silly::CmpBinOpKind::Equal, left, right, ls );
                    break;
                }
                case ExprOp::NotEqual:
                {
                    v = createBinaryCompare( vLoc, silly::CmpBinOpKind::NotEqual, left, right, ls );
                    break;
                }
                case ExprOp::Less:
                {
                    v = createBinaryCompare( vLoc, silly::CmpBinOpKind::Less, left, right, ls );
                    break;
                }
                case ExprOp::LessEqual:
                {
                    v = createBinaryCompare( vLoc, silly::CmpBinOpKind::LessEq, left, right, ls );
                    break;
                }
                case ExprOp::Greater:
                {
                    v = createBinaryCompare( vLoc, silly::CmpBinOpKind::Less, right, left, ls );
                    break;
                }
                case ExprOp::GreaterEqual:
                {
                    v = createBinaryCompare( vLoc, silly::CmpBinOpKind::LessEq, right, left, ls );
                    break;
                }
                default:
                    emitInternalError( vLoc, __FILE__, __LINE__, __func__,
                                       std::format( "parseExpression failed. Unknown binary op: {}", (int)parg.kind ),
                                       currentFuncName );
            }
        }
        else if ( parg.kind == Expr::Kind::Call )
        {
            v = generateCall( parg.name, parg.params, vLoc, false );
        }
        else
        {
            mlir::Value index;
            if ( parg.kind == Expr::Kind::ArrayVariable )
            {
                index = parseExpression( vLoc, {}, *parg.left, ls );
            }

            v = createVariableLoad( vLoc, parg.name, index, vLoc, ls );
        }

        return v;
    }

    void BisonParseListener::enterForStatement( const silly::BisonParser::location_type& bForLoc,
                                                const silly::Types& intType,
                                                const silly::BisonParser::location_type& bVarLoc,
                                                const std::string& varName, const silly::Expr& start,
                                                const silly::Expr& stop, const silly::Expr& step )
    {
        mlir::Location loc = getLocation( bForLoc );
        mlir::Location varLoc = getLocation( bVarLoc );
        mlir::Type elemType = declarationType( loc, intType );
        LocationStack ls( builder, loc );
        mlir::Value vstart = parseExpression( varLoc, elemType, start, ls );
        mlir::Value vstop = parseExpression( varLoc, elemType, stop, ls );
        mlir::Value vstep;
        if ( step.kind != silly::Expr::Kind::None )
        {
            vstep = parseExpression( varLoc, elemType, step, ls );
        }

        // checkForReturnInScope( ctx->scopedStatements(), "ELIF block" );

        createFor( loc, varName, elemType, varLoc, vstart, vstop, vstep, ls );
    }

    void BisonParseListener::exitForStatement( const silly::BisonParser::location_type& bForLoc )
    {
        mlir::Location loc = getLocation( bForLoc );

        finishFor( loc );
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
            mlir::Value v = parseExpression( vLoc, {}, parg, ls );
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

    mlir::Type BisonParseListener::declarationType( mlir::Location loc, const Types type )
    {
        mlir::Type ty{};

        switch ( type )
        {
            case Types::Boolean:
            {
                ty = typ.i1;
                break;
            }
            case Types::Int8:
            {
                ty = typ.i8;
                break;
            }
            case Types::Int16:
            {
                ty = typ.i16;
                break;
            }
            case Types::Int32:
            {
                ty = typ.i32;
                break;
            }
            case Types::Int64:
            {
                ty = typ.i64;
                break;
            }
            case Types::Float32:
            {
                ty = typ.f32;
                break;
            }
            case Types::Float64:
            {
                ty = typ.f64;
                break;
            }
            case Types::None:
            {
                break;
            }
            default:
                emitInternalError( loc, __FILE__, __LINE__, __func__, "Unsupported declaration type.",
                                   currentFuncName );
        }

        return ty;
    }

    void BisonParseListener::declarationHelper( mlir::Location tLoc, mlir::Location aLoc, const std::string& varName,
                                                const std::string& arraySizeString, mlir::Type ty, bool hasInit,
                                                const std::vector<silly::Expr>& initializerLiterals, LocationStack& ls )
    {
        std::vector<mlir::Value> initializers;

        for ( const silly::Expr& e : initializerLiterals )
        {
            mlir::Value init = parseExpression( tLoc, ty, e, ls );

            if ( !init )
            {
                emitInternalError( tLoc, __FILE__, __LINE__, __func__, "parseExpression failed", currentFuncName );
                return;
            }

            initializers.push_back( init );
        }

        createDeclaration( tLoc, varName, ty, aLoc, arraySizeString, hasInit, initializers, ls );
    }

    void BisonParseListener::enterDeclareStatement( const silly::Types& type, const std::string& varName,
                                                    const std::string& arraySizeString,
                                                    const std::vector<silly::Expr>& initializers,
                                                    const silly::BisonParser::location_type& typeLoc,
                                                    const silly::BisonParser::location_type& nameLoc,
                                                    const silly::BisonParser::location_type& arrayLoc )
    {
        mlir::Location tLoc = getLocation( typeLoc );
        mlir::Type ty = declarationType( tLoc, type );
        if ( !ty )
        {
            return;
        }
        mlir::Location aLoc = getLocation( arrayLoc );
        LocationStack ls( builder, tLoc );

        declarationHelper( tLoc, aLoc, varName, arraySizeString, ty, true, initializers, ls );
    }

    // Treat this differently than enterDeclareStatement w/ no initializers (which implies -init-fill memset),
    // whereas any initializer expression (even empty) means all array members that don't have
    // explicit init get a binary zero value.
    void BisonParseListener::enterDeclareStatementWithEmptyInit( const silly::Types& type, const std::string& varName,
                                                                 const std::string& arraySizeString,
                                                                 const silly::BisonParser::location_type& typeLoc,
                                                                 const silly::BisonParser::location_type& nameLoc,
                                                                 const silly::BisonParser::location_type& arrayLoc )
    {
        std::vector<silly::Expr> initializers;
        enterDeclareStatement( type, varName, arraySizeString, initializers, typeLoc, nameLoc, arrayLoc );
    }

    void BisonParseListener::enterDeclareStatement( const silly::Types& type, const std::string& varName,
                                                    const std::string& arraySizeString,
                                                    const silly::BisonParser::location_type& typeLoc,
                                                    const silly::BisonParser::location_type& nameLoc,
                                                    const silly::BisonParser::location_type& arrayLoc )
    {
        mlir::Location tLoc = getLocation( typeLoc );
        mlir::Type ty = declarationType( tLoc, type );
        if ( !ty )
        {
            return;
        }
        mlir::Location aLoc = getLocation( arrayLoc );
        LocationStack ls( builder, tLoc );
        std::vector<silly::Expr> initializers;

        declarationHelper( tLoc, aLoc, varName, arraySizeString, ty, false, initializers, ls );
    }

    void BisonParseListener::enterAssignmentStatement( const silly::Expr& var, const silly::Expr& rhs,
                                                       const silly::BisonParser::location_type& lhsLoc,
                                                       const silly::BisonParser::location_type& rhsLoc )
    {
        mlir::Location loc = getLocation( lhsLoc );
        LocationStack ls( builder, loc );

        bool declared = isDeclared( var.name );
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
            indexValue = parseExpression( rLoc, {}, *var.left, ls );
            if ( !indexValue )
            {
                emitInternalError( loc, __FILE__, __LINE__, __func__, "parseExpression failed", currentFuncName );
                return;
            }
        }

        mlir::Location aLoc = getLocation( rhsLoc );
        mlir::Value resultValue = parseExpression( aLoc, {}, rhs, ls );
        if ( !resultValue )
        {
            return;
        }

        createAssignment( aLoc, resultValue, var.name, indexValue, ls );
    }

    void BisonParseListener::enterGetStatement( const silly::BisonParser::location_type& bLoc,
                                                const std::string& varName )
    {
        mlir::Location loc = getLocation( bLoc );
        LocationStack ls( builder, loc );

        createGet( loc, varName, {}, loc, ls );
    }

    void BisonParseListener::enterGetStatement( const silly::BisonParser::location_type& bLoc,
                                                const std::string& varName, const silly::Expr& indexExpr )
    {
        mlir::Location loc = getLocation( bLoc );
        LocationStack ls( builder, loc );

        mlir::Location iloc = loc;    // FIXME.
        mlir::Value idx = parseExpression( loc, {}, indexExpr, ls );
        createGet( loc, varName, idx, iloc, ls );
    }

    void BisonParseListener::enterImportStatement( const silly::BisonParser::location_type& bLoc,
                                                   const std::string& modName )
    {
        mlir::Location loc = getLocation( bLoc );

        createImport( loc, modName );
    }

    void BisonParseListener::functionHelper( const std::string& name, const std::vector<silly::TypeAndName>& params,
                                             const silly::Types& returnType,
                                             const silly::BisonParser::location_type& funcLoc, bool isDeclaration )
    {
        LocPairs locs = getLocations( funcLoc );

        mlir::Type rt = declarationType( locs.first, returnType );

        std::vector<mlir::Type> paramTypes;
        std::vector<std::string> paramNames;
        for ( const silly::TypeAndName& tn : params )
        {
            mlir::Type paramType = declarationType( locs.first, tn.typ );
            paramTypes.push_back( paramType );
            paramNames.push_back( tn.name );
        }

        createFunction( locs, name, isDeclaration, rt, paramTypes, paramNames );
    }

    void BisonParseListener::enterFunctionPrototype( const std::string& name,
                                                     const std::vector<silly::TypeAndName>& params,
                                                     const silly::Types& returnType,
                                                     const silly::BisonParser::location_type& funcLoc )
    {
        functionHelper( name, params, returnType, funcLoc, true );

        finishFunction();
    }

    void BisonParseListener::enterFunctionDefinition( const std::string& name,
                                                      const std::vector<silly::TypeAndName>& params,
                                                      const silly::Types& returnType,
                                                      const silly::BisonParser::location_type& funcLoc )
    {
        functionHelper( name, params, returnType, funcLoc, false );
    }

    void BisonParseListener::exitFunctionDefinition()
    {
        finishFunction();
    }

    void BisonParseListener::enterReturnStatement( const silly::BisonParser::location_type& bLoc,
                                                   const silly::Expr& expr )
    {
        LocPairs locs = getLocations( bLoc );
        LocationStack ls( builder, locs.first );

        mlir::Value value;
        if ( expr.kind != silly::Expr::Kind::None )
        {
            value = parseExpression( locs.first, {}, expr, ls );
        }

        createReturn( locs.second, value, ls );
    }

    void BisonParseListener::enterIfStatement( const silly::BisonParser::location_type& bLoc,
                                               const silly::Expr& predicate )
    {
        mlir::Location loc = getLocation( bLoc );
        LocationStack ls( builder, loc );

        // checkForReturnInScope( ctx->scopedStatements(), "IF block" );

        mlir::Value conditionPredicate = parseExpression( loc, {}, predicate, ls );
        if ( !conditionPredicate )
        {
            emitInternalError( loc, __FILE__, __LINE__, __func__, "parseExpression failed", currentFuncName );
            return;
        }

        createIf( loc, conditionPredicate, true, ls );
    }

    void BisonParseListener::enterElifStatement( const silly::BisonParser::location_type& bLoc,
                                                 const silly::Expr& predicate )
    {
        mlir::Location loc = getLocation( bLoc );
        LocationStack ls( builder, loc );

        // checkForReturnInScope( ctx->scopedStatements(), "ELIF block" );

        selectElseBlock( loc );

        mlir::Value conditionPredicate = parseExpression( loc, {}, predicate, ls );
        if ( !conditionPredicate )
        {
            emitInternalError( loc, __FILE__, __LINE__, __func__, "parseExpression failed", currentFuncName );
            return;
        }

        createIf( loc, conditionPredicate, false, ls );
    }

    void BisonParseListener::enterElseStatement( const silly::BisonParser::location_type& bLoc )
    {
        mlir::Location loc = getLocation( bLoc );

        // checkForReturnInScope( ctx->scopedStatements(), "ELSE block" );

        selectElseBlock( loc );
    }

    void BisonParseListener::exitIfElifElseStatement( const silly::BisonParser::location_type& bLoc )
    {
        finishIfElifElse();
    }

    template <typename T>
    static const silly::Expr& derefExpr( const T& e )
    {
        return e;
    }

    template <>
    const silly::Expr& derefExpr( const std::shared_ptr<silly::Expr>& e )
    {
        return *e;
    }

    // This template parameterization is a hack.  Should probably switch to std::shared_ptr<silly::Expr> uniformly
    // instead.
    template <class ExprVector>
    mlir::Value BisonParseListener::generateCall( const std::string& name, const ExprVector& args, mlir::Location loc,
                                                  bool isCallStatement )
    {
        LocationStack ls( builder, loc );
        ParserPerFunctionState& f = lookupFunctionState( name );
        mlir::func::FuncOp funcOp = f.getFuncOp();
        mlir::FunctionType funcType = funcOp.getFunctionType();

        std::vector<mlir::Value> parameters;

        int i = 0;

        for ( const auto& e : args )
        {
            mlir::Type ty = funcType.getInputs()[i];
            mlir::Value value = parseExpression( loc, ty, derefExpr( e ), ls );
            if ( !value )
            {
                emitInternalError( loc, __FILE__, __LINE__, __func__, "parseExpression failed", currentFuncName );
                return value;
            }

            parameters.push_back( value );
            i++;
        }

        return createCall( loc, name, funcOp, funcType, isCallStatement, parameters, ls );
    }

    void BisonParseListener::enterCallStatement( const std::string& name, const std::vector<silly::Expr>& args,
                                                 const silly::BisonParser::location_type& bLoc )
    {
        mlir::Location loc = getLocation( bLoc );
        generateCall( name, args, loc, true );
    }

    void BisonParseListener::enterScopedStatements( const silly::BisonParser::location_type& bLoc )
    {
        mlir::Location loc = getLocation( bLoc );

        enterScopedRegion( loc, true );
    }

    void BisonParseListener::exitScopedStatements( )
    {
        exitScopedRegion( );
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
