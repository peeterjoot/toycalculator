///
/// @file    parser.cpp
/// @author  Peeter Joot <peeterjoot@pm.me>
/// @brief   altlr4 parse tree listener and MLIR builder.
///
#include <llvm/Support/Debug.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Types.h>

#include <format>

#include "SillyExceptions.hpp"
#include "parser.hpp"

#define DEBUG_TYPE "silly-parser"

#define ENTRY_SYMBOL_NAME "main"

#define CATCH_USER_ERROR                                \
    catch ( const UserError &e )                        \
    {                                                   \
        mlir::emitError( e.getLocation() ) << e.what(); \
        hasErrors = true;                               \
    }

namespace silly
{
    inline std::string mlirTypeToString( mlir::Type t )
    {
        std::string s;
        llvm::raw_string_ostream( s ) << t;
        return s;
    }

    inline PerFunctionState &MLIRListener::funcState( const std::string &funcName )
    {
        if ( !functionStateMap.contains( funcName ) )
        {
            functionStateMap[funcName] = std::make_unique<PerFunctionState>();
        }

        return *functionStateMap[funcName];
    }

    inline void MLIRListener::setFuncNameAndOp( const std::string &funcName, mlir::Operation *op )
    {
        currentFuncName = funcName;
        PerFunctionState &f = funcState( currentFuncName );
        f.funcOp = op;
    }

    inline std::string formatLocation( mlir::Location loc )
    {
        if ( mlir::FileLineColLoc fileLoc = mlir::dyn_cast<mlir::FileLineColLoc>( loc ) )
        {
            return std::format( "{}:{}:{}: ", fileLoc.getFilename().str(), fileLoc.getLine(), fileLoc.getColumn() );
        }
        return "";
    }

    inline mlir::Value MLIRListener::parseBoolean( mlir::Location loc, const std::string &s )
    {
        int val;
        if ( s == "TRUE" )
        {
            val = 1;
        }
        else if ( s == "FALSE" )
        {
            val = 0;
        }
        else
        {
            throw ExceptionWithContext( __FILE__, __LINE__, __func__,
                                        std::format( "{}error: internal error: boolean value neither TRUE nor "
                                                     "FALSE: {}.\n",
                                                     formatLocation( loc ), s ) );
        }

        return builder.create<mlir::arith::ConstantIntOp>( loc, val, 1 );
    }

    inline mlir::Value MLIRListener::parseInteger( mlir::Location loc, int width, const std::string &s )
    {
        int64_t val = std::stoll( s );

        return builder.create<mlir::arith::ConstantIntOp>( loc, val, width );
    }

    inline mlir::Value MLIRListener::parseFloat( mlir::Location loc, mlir::FloatType ty, const std::string &s )
    {
        if ( ty == tyF32 )
        {
            float val = std::stof( s );

            llvm::APFloat apVal( val );

            return builder.create<mlir::arith::ConstantFloatOp>( loc, tyF32, apVal );
        }
        else
        {
            double val = std::stod( s );

            llvm::APFloat apVal( val );

            return builder.create<mlir::arith::ConstantFloatOp>( loc, tyF64, apVal );
        }
    }

    inline mlir::func::FuncOp MLIRListener::getFuncOp( mlir::Location loc, const std::string &funcName )
    {
        PerFunctionState &f = funcState( funcName );
        mlir::func::FuncOp op = mlir::cast<mlir::func::FuncOp>( f.funcOp );

        if ( op == nullptr )
        {
            throw ExceptionWithContext( __FILE__, __LINE__, __func__,
                                        std::format( "{}internal error: Unable to find FuncOp for function {}\n",
                                                     formatLocation( loc ), funcName ) );
        }
        return op;
    }

    inline std::string MLIRListener::stripQuotes( mlir::Location loc, const std::string &input ) const
    {
        if ( ( input.size() < 2 ) || ( input.front() != '"' ) || ( input.back() != '"' ) )
        {
            throw ExceptionWithContext(
                __FILE__, __LINE__, __func__,
                std::format( "{}internal error: String '{}' was not double quotes enclosed as expected.\n",
                             formatLocation( loc ), input ) );
        }

        return input.substr( 1, input.size() - 2 );
    }

    inline LocPairs MLIRListener::getLocations( antlr4::ParserRuleContext *ctx )
    {
        size_t startLine = 1;
        size_t startCol = 0;
        size_t endLine = 1;
        size_t endCol = 0;

        if ( ctx )
        {
            antlr4::Token *startToken = ctx->getStart();
            startLine = startToken->getLine();
            startCol = startToken->getCharPositionInLine();

            antlr4::Token *endToken = ctx->getStop();
            endLine = endToken->getLine();
            endCol = endToken->getCharPositionInLine();
        }

        mlir::FileLineColLoc startLoc =
            mlir::FileLineColLoc::get( builder.getStringAttr( filename ), startLine, startCol + 1 );
        mlir::FileLineColLoc endLoc =
            mlir::FileLineColLoc::get( builder.getStringAttr( filename ), endLine, endCol + 1 );

        return { startLoc, endLoc };
    }

    inline mlir::Location MLIRListener::getStartLocation( antlr4::ParserRuleContext *ctx )
    {
        LocPairs locs = getLocations( ctx );

        return locs.first;
    }

    inline mlir::Location MLIRListener::getStopLocation( antlr4::ParserRuleContext *ctx )
    {
        LocPairs locs = getLocations( ctx );

        return locs.second;
    }

    DialectCtx::DialectCtx()
    {
        context.getOrLoadDialect<silly::SillyDialect>();
        context.getOrLoadDialect<mlir::func::FuncDialect>();
        context.getOrLoadDialect<mlir::arith::ArithDialect>();
        context.getOrLoadDialect<mlir::memref::MemRefDialect>();
        context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
        context.getOrLoadDialect<mlir::scf::SCFDialect>();
    }

    void MLIRListener::registerDeclaration( mlir::Location loc, const std::string &varName, mlir::Type ty,
                                            SillyParser::ArrayBoundsExpressionContext *arrayBounds,
                                            std::vector<SillyParser::BooleanLiteralContext *> *booleanLiteral,
                                            std::vector<SillyParser::IntegerLiteralContext *> *integerLiteral,
                                            std::vector<SillyParser::NumericLiteralContext *> *numericLiteral )
    {
        size_t arraySize{};
        size_t numElements{ 1 };
        if ( arrayBounds )
        {
            tNode *index = arrayBounds->INTEGER_PATTERN();
            assert( index );
            arraySize = std::stoi( index->getText() );
            numElements = arraySize;
        }

        mlir::OpBuilder::InsertPoint savedIP = builder.saveInsertionPoint();

        // Get the single scope
        mlir::func::FuncOp funcOp = getFuncOp( loc, currentFuncName );

        silly::ScopeOp scopeOp = getEnclosingScopeOp( loc, funcOp );

        mlir::Operation *symbolOp = mlir::SymbolTable::lookupSymbolIn( scopeOp, varName );
        if ( symbolOp )
        {
            throw UserError( loc, std::format( "Variable {} already declared", varName ) );
        }

        // Scope has one block
        mlir::Block *scopeBlock = &scopeOp.getBody().front();

        // Insert declarations at the beginning of the scope block
        // (all DeclareOps should appear before any scf.if/scf.for)
        builder.setInsertionPointToStart( scopeBlock );

        std::vector<mlir::Value> initializers;

        if ( booleanLiteral )
        {
            assert( integerLiteral == nullptr );
            assert( numericLiteral == nullptr );
            for ( SillyParser::BooleanLiteralContext *lit : *booleanLiteral )
            {
                if ( tNode *b = lit->BOOLEAN_PATTERN() )
                {
                    initializers.push_back( parseBoolean( loc, b->getText() ) );
                }
                else
                {
                    throw ExceptionWithContext(
                        __FILE__, __LINE__, __func__,
                        std::format( "{}internal error: boolean literal with unknown type\n", formatLocation( loc ) ) );
                }
            }

            ssize_t remaining = numElements - initializers.size();
            mlir::Value fill{};
            for ( ssize_t i = 0; i < remaining; i++ )
            {
                if ( i == 0 )
                {
                    fill = parseBoolean( loc, "FALSE" );
                }

                initializers.push_back( fill );
            }
        }
        else if ( integerLiteral )
        {
            assert( booleanLiteral == nullptr );
            assert( numericLiteral == nullptr );

            mlir::IntegerType ity = mlir::cast<mlir::IntegerType>( ty );
            int width = ity.getWidth();

            for ( SillyParser::IntegerLiteralContext *lit : *integerLiteral )
            {
                if ( tNode *i = lit->INTEGER_PATTERN() )
                {
                    initializers.push_back( parseInteger( loc, width, i->getText() ) );
                }
                else
                {
                    throw ExceptionWithContext(
                        __FILE__, __LINE__, __func__,
                        std::format( "{}internal error: integer literal with unknown type\n", formatLocation( loc ) ) );
                }
            }

            ssize_t remaining = numElements - initializers.size();
            mlir::Value fill{};
            for ( ssize_t i = 0; i < remaining; i++ )
            {
                if ( i == 0 )
                {
                    fill = parseInteger( loc, width, "0" );
                }

                initializers.push_back( fill );
            }
        }
        else if ( numericLiteral )
        {
            assert( booleanLiteral == nullptr );
            assert( integerLiteral == nullptr );

            mlir::FloatType fty = mlir::cast<mlir::FloatType>( ty );

            for ( SillyParser::NumericLiteralContext *lit : *numericLiteral )
            {
                if ( tNode *i = lit->INTEGER_PATTERN() )
                {
                    initializers.push_back( parseFloat( loc, fty, i->getText() ) );
                }
                else if ( tNode *f = lit->FLOAT_PATTERN() )
                {
                    initializers.push_back( parseFloat( loc, fty, f->getText() ) );
                }
                else
                {
                    throw ExceptionWithContext(
                        __FILE__, __LINE__, __func__,
                        std::format( "{}internal error: floating point literal with unknown type\n",
                                     formatLocation( loc ) ) );
                }
            }

            ssize_t remaining = numElements - initializers.size();
            mlir::Value fill{};
            for ( ssize_t i = 0; i < remaining; i++ )
            {
                if ( i == 0 )
                {
                    fill = parseFloat( loc, fty, "0" );
                }

                initializers.push_back( fill );
            }
        }

        if ( initializers.size() > numElements )
        {
            throw UserError(
                loc, std::format( "For variable '{}', more initializers ({}) specified than number of elements ({}).\n",
                                  varName, initializers.size(), numElements ) );
        }

        mlir::StringAttr strAttr = builder.getStringAttr( varName );
        silly::DeclareOp dcl;
        if ( arraySize )
        {
            dcl = builder.create<silly::DeclareOp>( loc, mlir::TypeAttr::get( ty ),
                                                    builder.getI64IntegerAttr( arraySize ),
                                                    /*parameter=*/nullptr, nullptr, initializers );
        }
        else
        {
            dcl = builder.create<silly::DeclareOp>( loc, mlir::TypeAttr::get( ty ), nullptr, /*parameter=*/nullptr,
                                                    nullptr, initializers );
        }
        dcl->setAttr( "sym_name", strAttr );

        builder.restoreInsertionPoint( savedIP );
    }

    inline mlir::Value MLIRListener::parseExpression( mlir::Location loc, SillyParser::ExpressionContext *ctx,
                                                      mlir::Type opType )
    {
        return parseLogicalOr( loc, ctx, opType );
    }

    inline mlir::Value MLIRListener::parseRvalue( mlir::Location loc, SillyParser::RvalueExpressionContext *ctx,
                                                  mlir::Type opType )
    {
        assert( ctx && ctx->expression() );

        return parseExpression( loc, ctx->expression(), opType );
    }

    void MLIRListener::syntaxError( antlr4::Recognizer *recognizer, antlr4::Token *offendingSymbol, size_t line,
                                    size_t charPositionInLine, const std::string &msg, std::exception_ptr e )
    {
        hasErrors = true;
        std::string tokenText = offendingSymbol ? offendingSymbol->getText() : "<none>";
        throw ExceptionWithContext( __FILE__, __LINE__, __func__,
                                    std::format( "Syntax error in {}:{}:{}: {} (token: {} )", filename, line,
                                                 charPositionInLine, msg, tokenText ) );
    }

    silly::ScopeOp getEnclosingScopeOp( mlir::Location loc, mlir::func::FuncOp funcOp )
    {
        // Single ScopeOp per function – iterate once to find it
        for ( mlir::Operation &op : funcOp.getBody().front() )
        {
            if ( silly::ScopeOp scopeOp = dyn_cast<silly::ScopeOp>( &op ) )
            {
                return scopeOp;
            }
        }

        throw ExceptionWithContext(
            __FILE__, __LINE__, __func__,
            std::format( "{}internal error: Unable to find Enclosing ScopeOp\n", formatLocation( loc ) ) );

        return nullptr;
    }

    silly::DeclareOp MLIRListener::lookupDeclareForVar( mlir::Location loc, const std::string &varName )
    {
        mlir::func::FuncOp funcOp = getFuncOp( loc, currentFuncName );

        silly::ScopeOp scopeOp = getEnclosingScopeOp( loc, funcOp );

        LLVM_DEBUG( {
            llvm::errs() << std::format( "Lookup symbol {} in parent function:\n", varName );
            scopeOp->dump();
        } );

        mlir::Operation *symbolOp = mlir::SymbolTable::lookupSymbolIn( scopeOp, varName );
        if ( !symbolOp )
        {
            throw UserError( loc, std::format( "Undeclared variable {} (symbol lookup failed.)", varName ) );
        }

        silly::DeclareOp declareOp = mlir::dyn_cast<silly::DeclareOp>( symbolOp );
        if ( !declareOp )
        {
            throw UserError( loc, std::format( "Undeclared variable {} (DeclareOp not found)", varName ) );
        }

        return declareOp;
    }

    mlir::Type MLIRListener::parseScalarType( const std::string &ty )
    {
        if ( ty == "BOOL" )
        {
            return tyI1;
        }
        if ( ty == "INT8" )
        {
            return tyI8;
        }
        if ( ty == "INT16" )
        {
            return tyI16;
        }
        if ( ty == "INT32" )
        {
            return tyI32;
        }
        if ( ty == "INT64" )
        {
            return tyI64;
        }
        if ( ty == "FLOAT32" )
        {
            return tyF32;
        }
        if ( ty == "FLOAT64" )
        {
            return tyF64;
        }
        return nullptr;
    }

    MLIRListener::MLIRListener( const std::string &filenameIn )
        : filename( filenameIn ),
          dialect(),
          builder( &dialect.context ),
          mod( mlir::ModuleOp::create( getStartLocation( nullptr ) ) )
    {
        builder.setInsertionPointToStart( mod.getBody() );

        tyI1 = builder.getI1Type();
        tyI8 = builder.getI8Type();
        tyI16 = builder.getI16Type();
        tyI32 = builder.getI32Type();
        tyI64 = builder.getI64Type();

        tyF32 = builder.getF32Type();
        tyF64 = builder.getF64Type();

        mlir::MLIRContext *ctx = builder.getContext();
        tyVoid = mlir::LLVM::LLVMVoidType::get( ctx );
        tyPtr = mlir::LLVM::LLVMPointerType::get( ctx );
    }

    void MLIRListener::createScope( mlir::Location startLoc, mlir::Location endLoc, mlir::func::FuncOp funcOp,
                                    const std::string &funcName, const std::vector<std::string> &paramNames )
    {
        LLVM_DEBUG( {
            llvm::errs() << std::format( "createScope: {}: startLoc: {}, endLoc: {}\n", funcName,
                                         formatLocation( startLoc ), formatLocation( endLoc ) );
        } );
        mlir::Block &block = *funcOp.addEntryBlock();
        builder.setInsertionPointToStart( &block );

        // initially with empty operands and results
        silly::ScopeOp scopeOp = builder.create<silly::ScopeOp>( startLoc, mlir::TypeRange{}, mlir::ValueRange{} );
        builder.create<silly::YieldOp>( endLoc );

        mlir::Block &scopeBlock = scopeOp.getBody().emplaceBlock();

        builder.setInsertionPointToStart( &scopeBlock );

        for ( size_t i = 0; i < funcOp.getNumArguments() && i < paramNames.size(); ++i )
        {
            mlir::Type argType = funcOp.getArgument( i ).getType();
            LLVM_DEBUG( {
                llvm::errs() << std::format( "function {}: parameter{}:\n", funcName, i );
                funcOp.getArgument( i ).dump();
            } );
            mlir::StringAttr strAttr = builder.getStringAttr( paramNames[i] );
            silly::DeclareOp dcl = builder.create<silly::DeclareOp>(
                startLoc, mlir::TypeAttr::get( argType ), /*size=*/nullptr, builder.getUnitAttr(),
                builder.getI64IntegerAttr( i ), mlir::ValueRange{} );
            dcl->setAttr( "sym_name", strAttr );
        }

        setFuncNameAndOp( funcName, funcOp );
    }

    void MLIRListener::enterStartRule( SillyParser::StartRuleContext *ctx )
    try
    {
        assert( ctx );
        currentFuncName = ENTRY_SYMBOL_NAME;

        LocPairs locs = getLocations( ctx );

        mlir::FunctionType funcType = builder.getFunctionType( {}, tyI32 );
        mlir::func::FuncOp funcOp = builder.create<mlir::func::FuncOp>( locs.first, ENTRY_SYMBOL_NAME, funcType );

        std::vector<std::string> paramNames;
        createScope( locs.first, locs.second, funcOp, ENTRY_SYMBOL_NAME, paramNames );
    }
    CATCH_USER_ERROR

    void MLIRListener::processReturnLike( mlir::Location loc, SillyParser::RvalueExpressionContext *rvalueExpression )
    {
        mlir::Type returnType{};
        mlir::Value value{};

        if ( currentFuncName == ENTRY_SYMBOL_NAME )
        {
            returnType = tyI32;
        }
        else
        {
            mlir::func::FuncOp func = getFuncOp( loc, currentFuncName );
            llvm::ArrayRef<mlir::Type> returnTypeArray = func.getFunctionType().getResults();

            if ( !returnTypeArray.empty() )
            {
                returnType = returnTypeArray[0];
            }
        }

        if ( rvalueExpression )
        {
            value = parseRvalue( loc, rvalueExpression, returnType );

            value = castOpIfRequired( loc, value, returnType );
        }
        else if ( currentFuncName == ENTRY_SYMBOL_NAME )
        {
            value = builder.create<mlir::arith::ConstantIntOp>( loc, 0, 32 );
        }

        // Create new ReturnOp with user specified value:
        if ( value )
        {
            builder.create<silly::ReturnOp>( loc, mlir::ValueRange{ value } );
        }
        else
        {
            builder.create<silly::ReturnOp>( loc, mlir::ValueRange{} );
        }
    }

    void MLIRListener::exitStartRule( SillyParser::StartRuleContext *ctx )
    try
    {
        assert( ctx );
        mlir::Location loc = getStartLocation( ctx );

        assert( currentFuncName == ENTRY_SYMBOL_NAME );

        if ( !ctx->exitStatement() )
        {
            processReturnLike( loc, nullptr );
        }

        LLVM_DEBUG( {
            llvm::errs() << "exitStartRule done: module dump:\n";
            mod->dump();
        } );
    }
    CATCH_USER_ERROR

    void MLIRListener::enterFunction( SillyParser::FunctionContext *ctx )
    try
    {
        assert( ctx );
        LocPairs locs = getLocations( ctx );

        LLVM_DEBUG( {
            llvm::errs() << std::format( "enterFunction: startLoc: {}, endLoc: {}:\n", formatLocation( locs.first ),
                                         formatLocation( locs.second ) );
        } );

        mainIP = builder.saveInsertionPoint();

        builder.setInsertionPointToStart( mod.getBody() );

        assert( ctx->IDENTIFIER() );
        std::string funcName = ctx->IDENTIFIER()->getText();

        std::vector<mlir::Type> returns;
        if ( SillyParser::ScalarTypeContext *rt = ctx->scalarType() )
        {
            mlir::Type returnType = parseScalarType( rt->getText() );
            returns.push_back( returnType );
        }

        std::vector<mlir::Type> paramTypes;
        std::vector<std::string> paramNames;
        for ( SillyParser::VariableTypeAndNameContext *paramCtx : ctx->variableTypeAndName() )
        {
            assert( paramCtx->scalarType() );
            mlir::Type paramType = parseScalarType( paramCtx->scalarType()->getText() );
            assert( paramCtx->IDENTIFIER() );
            std::string paramName = paramCtx->IDENTIFIER()->getText();
            paramTypes.push_back( paramType );
            paramNames.push_back( paramName );
        }

        std::vector<mlir::NamedAttribute> attrs;
        attrs.push_back(
            mlir::NamedAttribute( builder.getStringAttr( "sym_visibility" ), builder.getStringAttr( "private" ) ) );

        mlir::FunctionType funcType = builder.getFunctionType( paramTypes, returns );
        mlir::func::FuncOp funcOp = builder.create<mlir::func::FuncOp>( locs.first, funcName, funcType, attrs );
        createScope( locs.first, locs.second, funcOp, funcName, paramNames );
    }
    CATCH_USER_ERROR

    void MLIRListener::exitFunction( SillyParser::FunctionContext *ctx )
    try
    {
        assert( ctx );
        builder.restoreInsertionPoint( mainIP );

        currentFuncName = ENTRY_SYMBOL_NAME;
    }
    CATCH_USER_ERROR

    mlir::Value MLIRListener::handleCall( SillyParser::CallExpressionContext *ctx )
    {
        assert( ctx );
        mlir::Location loc = getStartLocation( ctx );
        tNode *id = ctx->IDENTIFIER();
        assert( id );
        std::string funcName = id->getText();
        mlir::func::FuncOp funcOp = getFuncOp( loc, funcName );
        mlir::FunctionType funcType = funcOp.getFunctionType();
        std::vector<mlir::Value> parameters;

        if ( SillyParser::ParameterListContext *params = ctx->parameterList() )
        {
            int i = 0;

            assert( params );
            size_t psz = params->parameterExpression().size();
            size_t fsz = funcType.getInputs().size();
            assert( psz == fsz );

            for ( SillyParser::ParameterExpressionContext *e : params->parameterExpression() )
            {
                SillyParser::RvalueExpressionContext *p = e->rvalueExpression();
                assert( p );
                std::string paramText = p->getText();
                std::cout << std::format( "CALL function {}: param: {}\n", funcName, paramText );

                mlir::Type ty = funcType.getInputs()[i];
                mlir::Value value = parseRvalue( loc, p, ty );
                value = castOpIfRequired( loc, value, ty );

                parameters.push_back( value );
                i++;
            }
        }

        mlir::TypeRange resultTypes = funcType.getResults();

        silly::CallOp callOp = builder.create<silly::CallOp>( loc, resultTypes, funcName, parameters );

        // Return the first result (or null for void calls)
        return resultTypes.empty() ? mlir::Value{} : callOp.getResults()[0];
    }

    void MLIRListener::enterCallStatement( SillyParser::CallStatementContext *ctx )
    try
    {
        assert( ctx );
        handleCall( ctx->callExpression() );
    }
    CATCH_USER_ERROR

    void MLIRListener::enterDeclare( SillyParser::DeclareContext *ctx )
    try
    {
        assert( ctx );
        mlir::Location loc = getStartLocation( ctx );
        assert( ctx->IDENTIFIER() );
        std::string varName = ctx->IDENTIFIER()->getText();

        std::vector<SillyParser::NumericLiteralContext *> *pNumericLiteral{};
        std::vector<SillyParser::NumericLiteralContext *> numericLiteral;
        if ( ctx->LEFT_CURLY_BRACKET_TOKEN() )
        {
            numericLiteral = ctx->numericLiteral();
            pNumericLiteral = &numericLiteral;
        }
        registerDeclaration( loc, varName, tyF64, ctx->arrayBoundsExpression(), nullptr, nullptr, pNumericLiteral );

        if ( SillyParser::AssignmentRvalueContext *expr = ctx->assignmentRvalue() )
        {
            processAssignment( loc, expr->rvalueExpression(), varName, {} );
        }
    }
    CATCH_USER_ERROR

    void MLIRListener::enterBoolDeclare( SillyParser::BoolDeclareContext *ctx )
    try
    {
        assert( ctx );
        mlir::Location loc = getStartLocation( ctx );
        std::string varName = ctx->IDENTIFIER()->getText();

        std::vector<SillyParser::BooleanLiteralContext *> *pBooleanLiteral{};
        std::vector<SillyParser::BooleanLiteralContext *> booleanLiteral;
        if ( ctx->LEFT_CURLY_BRACKET_TOKEN() )
        {
            booleanLiteral = ctx->booleanLiteral();
            pBooleanLiteral = &booleanLiteral;
        }
        registerDeclaration( loc, varName, tyI1, ctx->arrayBoundsExpression(), pBooleanLiteral, nullptr, nullptr );

        if ( SillyParser::AssignmentRvalueContext *expr = ctx->assignmentRvalue() )
        {
            processAssignment( loc, expr->rvalueExpression(), varName, {} );
        }
    }
    CATCH_USER_ERROR

    void MLIRListener::enterIntDeclare( SillyParser::IntDeclareContext *ctx )
    try
    {
        assert( ctx );
        mlir::Location loc = getStartLocation( ctx );
        assert( ctx->IDENTIFIER() );
        std::string varName = ctx->IDENTIFIER()->getText();

        mlir::Type ty;
        std::vector<SillyParser::IntegerLiteralContext *> *pIntegerLiteral{};
        std::vector<SillyParser::IntegerLiteralContext *> integerLiteral;
        if ( ctx->LEFT_CURLY_BRACKET_TOKEN() )
        {
            integerLiteral = ctx->integerLiteral();
            pIntegerLiteral = &integerLiteral;
        }

        if ( ctx->INT8_TOKEN() )
        {
            ty = tyI8;
        }
        else if ( ctx->INT16_TOKEN() )
        {
            ty = tyI16;
        }
        else if ( ctx->INT32_TOKEN() )
        {
            ty = tyI32;
        }
        else if ( ctx->INT64_TOKEN() )
        {
            ty = tyI64;
        }
        else
        {
            throw ExceptionWithContext( __FILE__, __LINE__, __func__,
                                        std::format( "{}internal error: Unsupported signed integer declaration size.\n",
                                                     formatLocation( loc ) ) );
        }

        registerDeclaration( loc, varName, ty, ctx->arrayBoundsExpression(), nullptr, pIntegerLiteral, nullptr );

        if ( SillyParser::AssignmentRvalueContext *expr = ctx->assignmentRvalue() )
        {
            processAssignment( loc, expr->rvalueExpression(), varName, {} );
        }
    }
    CATCH_USER_ERROR

    void MLIRListener::enterFloatDeclare( SillyParser::FloatDeclareContext *ctx )
    try
    {
        assert( ctx );
        mlir::Location loc = getStartLocation( ctx );
        assert( ctx->IDENTIFIER() );
        std::string varName = ctx->IDENTIFIER()->getText();
        mlir::Type ty;

        if ( ctx->FLOAT32_TOKEN() )
        {
            ty = tyF32;
        }
        else if ( ctx->FLOAT64_TOKEN() )
        {
            ty = tyF64;
        }
        else
        {
            throw ExceptionWithContext( __FILE__, __LINE__, __func__,
                                        std::format( "{}internal error: Unsupported floating point declaration size.\n",
                                                     formatLocation( loc ) ) );
        }

        std::vector<SillyParser::NumericLiteralContext *> *pNumericLiteral{};
        std::vector<SillyParser::NumericLiteralContext *> numericLiteral;
        if ( ctx->LEFT_CURLY_BRACKET_TOKEN() )
        {
            numericLiteral = ctx->numericLiteral();
            pNumericLiteral = &numericLiteral;
        }

        registerDeclaration( loc, varName, ty, ctx->arrayBoundsExpression(), nullptr, nullptr, pNumericLiteral );

        if ( SillyParser::AssignmentRvalueContext *expr = ctx->assignmentRvalue() )
        {
            processAssignment( loc, expr->rvalueExpression(), varName, {} );
        }
    }
    CATCH_USER_ERROR

    void MLIRListener::enterStringDeclare( SillyParser::StringDeclareContext *ctx )
    try
    {
        assert( ctx );
        mlir::Location loc = getStartLocation( ctx );
        assert( ctx->IDENTIFIER() );
        std::string varName = ctx->IDENTIFIER()->getText();
        SillyParser::ArrayBoundsExpressionContext *arrayBounds = ctx->arrayBoundsExpression();
        assert( arrayBounds );

        registerDeclaration( loc, varName, tyI8, arrayBounds, nullptr, nullptr, nullptr );

        if ( tNode *theString = ctx->STRING_PATTERN() )
        {
            std::string s = stripQuotes( loc, theString->getText() );

            mlir::SymbolRefAttr symRef = mlir::SymbolRefAttr::get( &dialect.context, varName );
            mlir::StringAttr strAttr = builder.getStringAttr( s );

            silly::StringLiteralOp stringLiteral = builder.create<silly::StringLiteralOp>( loc, tyPtr, strAttr );

            mlir::NamedAttribute varNameAttr( builder.getStringAttr( "var_name" ), symRef );

            builder.create<silly::AssignOp>( loc, mlir::TypeRange{}, mlir::ValueRange{ stringLiteral },
                                             llvm::ArrayRef<mlir::NamedAttribute>{ varNameAttr } );
        }
    }
    CATCH_USER_ERROR

    void MLIRListener::createIf( mlir::Location loc, SillyParser::BooleanValueContext *booleanValue, bool saveIP )
    {
        SillyParser::BoolAsExprContext *bc = dynamic_cast<SillyParser::BoolAsExprContext *>( booleanValue );

        mlir::Value conditionPredicate = parseExpression( loc, bc->expression(), tyI1 );

        if ( saveIP )
        {
            insertionPointStack.push_back( builder.saveInsertionPoint() );
        }

        mlir::scf::IfOp ifOp = builder.create<mlir::scf::IfOp>( loc, conditionPredicate,
                                                                /*withElseRegion=*/true );

        mlir::Block &thenBlock = ifOp.getThenRegion().front();
        builder.setInsertionPointToStart( &thenBlock );
    }

    void MLIRListener::enterIfStatement( SillyParser::IfStatementContext *ctx )
    try
    {
        assert( ctx );
        mlir::Location loc = getStartLocation( ctx );

        SillyParser::BooleanValueContext *booleanValue = ctx->booleanValue();
        assert( booleanValue );

        createIf( loc, booleanValue, true );
    }
    CATCH_USER_ERROR

    void MLIRListener::selectElseBlock( mlir::Location loc, const std::string &errorText )
    {
        mlir::Block *currentBlock = builder.getInsertionBlock();
        assert( currentBlock );

        // Get the parent region of the current block (the then region).
        mlir::Region *parentRegion = currentBlock->getParent();

        // Verify it's inside an scf.if by checking the parent op.
        mlir::Operation *parentOp = parentRegion->getParentOp();
        mlir::scf::IfOp ifOp = dyn_cast<mlir::scf::IfOp>( parentOp );

        if ( !ifOp )
        {
            throw ExceptionWithContext(
                __FILE__, __LINE__, __func__,
                std::format( "{}internal error: Current insertion point must be inside an scf.if then region\n",
                             formatLocation( loc ), errorText ) );
        }

        // Set the insertion point to the start of the else region's (first) block.
        mlir::Region &elseRegion = ifOp.getElseRegion();
        mlir::Block &elseBlock = elseRegion.front();
        builder.setInsertionPointToStart( &elseBlock );
    }

    void MLIRListener::enterElseStatement( SillyParser::ElseStatementContext *ctx )
    try
    {
        assert( ctx );
        mlir::Location loc = getStartLocation( ctx );

        selectElseBlock( loc, ctx->getText() );
    }
    CATCH_USER_ERROR

    void MLIRListener::enterElifStatement( SillyParser::ElifStatementContext *ctx )
    try
    {
        assert( ctx );
        mlir::Location loc = getStartLocation( ctx );

        selectElseBlock( loc, ctx->getText() );

        SillyParser::BooleanValueContext *booleanValue = ctx->booleanValue();

        createIf( loc, booleanValue, false );
    }
    CATCH_USER_ERROR

    void MLIRListener::exitIfelifelse( SillyParser::IfelifelseContext *ctx )
    try
    {
        // Restore EXACTLY where we were before creating the scf.if
        // This places new ops right AFTER the scf.if
        builder.restoreInsertionPoint( insertionPointStack.back() );
        insertionPointStack.pop_back();
    }
    CATCH_USER_ERROR

    void MLIRListener::enterFor( SillyParser::ForContext *ctx )
    try
    {
        assert( ctx );
        mlir::Location loc = getStartLocation( ctx );

        LLVM_DEBUG( { llvm::errs() << std::format( "For: {}\n", ctx->getText() ); } );

        assert( ctx->IDENTIFIER() );
        std::string varName = ctx->IDENTIFIER()->getText();

        mlir::SymbolRefAttr symRef = mlir::SymbolRefAttr::get( &dialect.context, varName );
        mlir::NamedAttribute varNameAttr( builder.getStringAttr( "var_name" ), symRef );

        assert( ctx->forStart() );
        assert( ctx->forEnd() );
        assert( ctx->forStart()->forRangeExpression() );
        assert( ctx->forEnd()->forRangeExpression() );
        SillyParser::RvalueExpressionContext *pStart = ctx->forStart()->forRangeExpression()->rvalueExpression();
        SillyParser::RvalueExpressionContext *pEnd = ctx->forEnd()->forRangeExpression()->rvalueExpression();
        SillyParser::RvalueExpressionContext *pStep{};
        if ( SillyParser::ForStepContext *st = ctx->forStep() )
        {
            assert( st->forRangeExpression() );
            pStep = st->forRangeExpression()->rvalueExpression();
        }

        mlir::Value start;
        mlir::Value end;
        mlir::Value step;

        silly::DeclareOp declareOp = lookupDeclareForVar( loc, varName );
        mlir::Type elemType = declareOp.getTypeAttr().getValue();

        std::string s;
        if ( pStart )
        {
            start = parseRvalue( loc, pStart, elemType );

            start = castOpIfRequired( loc, start, elemType );
        }
        else
        {
            throw ExceptionWithContext( __FILE__, __LINE__, __func__,
                                        std::format( "{}internal error: FOR loop: expected start index: {}\n",
                                                     formatLocation( loc ), ctx->getText() ) );
        }

        if ( pEnd )
        {
            end = parseRvalue( loc, pEnd, elemType );

            end = castOpIfRequired( loc, end, elemType );
        }
        else
        {
            throw ExceptionWithContext( __FILE__, __LINE__, __func__,
                                        std::format( "{}internal error: FOR loop: expected end index: {}\n",
                                                     formatLocation( loc ), ctx->getText() ) );
        }

        if ( pStep )
        {
            step = parseRvalue( loc, pStep, elemType );
        }
        else
        {
            //'scf.for' op failed to verify that all of {lowerBound, upperBound, step} have same type
            step = builder.create<mlir::arith::ConstantIntOp>( loc, 1, 64 );
        }

        step = castOpIfRequired( loc, step, elemType );

        insertionPointStack.push_back( builder.saveInsertionPoint() );

        mlir::scf::ForOp forOp = builder.create<mlir::scf::ForOp>( loc, start, end, step );

        mlir::Block &loopBody = forOp.getRegion().front();
        builder.setInsertionPointToStart( &loopBody );

        // emit an assignment to the variable as the first statement in the loop body, so that any existing references
        // to that will work as-is:
        mlir::Value inductionVar = loopBody.getArgument( 0 );
        builder.create<silly::AssignOp>( loc, mlir::TypeRange{}, mlir::ValueRange{ inductionVar },
                                         llvm::ArrayRef<mlir::NamedAttribute>{ varNameAttr } );
    }
    CATCH_USER_ERROR

    void MLIRListener::exitFor( SillyParser::ForContext *ctx )
    try
    {
        assert( ctx );
        builder.restoreInsertionPoint( insertionPointStack.back() );
        insertionPointStack.pop_back();
    }
    CATCH_USER_ERROR

    void MLIRListener::handlePrint( mlir::Location loc, const std::vector<SillyParser::RvalueExpressionContext *> &args,
                                    const std::string &errorContextString, PrintFlags pf )
    {
        std::vector<mlir::Value> vargs;
        for ( SillyParser::RvalueExpressionContext *parg : args )
        {
            mlir::Value v = parseRvalue( loc, parg, {} );

            vargs.push_back( v );
        }

        mlir::arith::ConstantIntOp constFlagOp = builder.create<mlir::arith::ConstantIntOp>( loc, pf, 32 );
        builder.create<silly::PrintOp>( loc, constFlagOp, vargs );
    }

    void MLIRListener::enterPrint( SillyParser::PrintContext *ctx )
    try
    {
        assert( ctx );
        mlir::Location loc = getStartLocation( ctx );

        int flags = PRINT_FLAGS_NONE;
        if ( ctx->CONTINUE_TOKEN() )
        {
            flags = PRINT_FLAGS_CONTINUE;
        }
        handlePrint( loc, ctx->rvalueExpression(), ctx->getText(), (PrintFlags)flags );
    }
    CATCH_USER_ERROR

    void MLIRListener::enterError( SillyParser::ErrorContext *ctx )
    try
    {
        assert( ctx );
        mlir::Location loc = getStartLocation( ctx );
        int flags = PRINT_FLAGS_ERROR;
        if ( ctx->CONTINUE_TOKEN() )
        {
            flags |= PRINT_FLAGS_CONTINUE;
        }
        handlePrint( loc, ctx->rvalueExpression(), ctx->getText(), (PrintFlags)flags );
    }
    CATCH_USER_ERROR

    void MLIRListener::enterAbort( SillyParser::AbortContext *ctx )
    try
    {
        assert( ctx );
        mlir::Location loc = getStartLocation( ctx );

        builder.create<silly::AbortOp>( loc );
    }
    CATCH_USER_ERROR

    void MLIRListener::enterGet( SillyParser::GetContext *ctx )
    try
    {
        assert( ctx );
        mlir::Location loc = getStartLocation( ctx );

        SillyParser::ScalarOrArrayElementContext *scalarOrArrayElement = ctx->scalarOrArrayElement();
        if ( scalarOrArrayElement )
        {
            tNode *varNameObject = scalarOrArrayElement->IDENTIFIER();
            assert( varNameObject );
            std::string varName = varNameObject->getText();

            silly::DeclareOp declareOp = lookupDeclareForVar( loc, varName );

            mlir::Type elemType = declareOp.getTypeAttr().getValue();
            mlir::Value optIndexValue{};
            if ( SillyParser::IndexExpressionContext *indexExpr = scalarOrArrayElement->indexExpression() )
            {
                mlir::Value indexValue = parseRvalue( loc, indexExpr->rvalueExpression(), elemType );

                optIndexValue = indexTypeCast( loc, indexValue );
            }
            else if ( declareOp.getSizeAttr() )
            {
                throw UserError( loc, std::format( "Attempted GET to string literal?: {}", ctx->getText() ) );
            }
            else
            {
                // Scalar: load the value
            }

            mlir::SymbolRefAttr symRef = mlir::SymbolRefAttr::get( &dialect.context, varName );

            silly::GetOp resultValue = builder.create<silly::GetOp>( loc, elemType );
            builder.create<silly::AssignOp>( loc, symRef, optIndexValue, resultValue );
        }
        else
        {
            throw ExceptionWithContext(
                __FILE__, __LINE__, __func__,
                std::format( "{}internal error: unexpected get context {}\n", formatLocation( loc ), ctx->getText() ) );
        }
    }
    CATCH_USER_ERROR

    mlir::Type MLIRListener::biggestTypeOf( mlir::Type ty1, mlir::Type ty2 )
    {
        if ( ty1 == ty2 )
        {
            return ty1;
        }
        else if ( ty1.isF64() )
        {
            return ty1;
        }
        else if ( ty2.isF64() )
        {
            return ty2;
        }
        else if ( ty1.isF32() )
        {
            return ty1;
        }
        else if ( ty2.isF32() )
        {
            return ty2;
        }
        else
        {
            mlir::IntegerType ity1 = mlir::cast<mlir::IntegerType>( ty1 );
            mlir::IntegerType ity2 = mlir::cast<mlir::IntegerType>( ty2 );

            unsigned w1 = ity1.getWidth();
            unsigned w2 = ity2.getWidth();

            if ( w1 > w2 )
            {
                return ty1;
            }
            else
            {
                return ty2;
            }
        }
    }

    mlir::Value MLIRListener::castOpIfRequired( mlir::Location loc, mlir::Value value, mlir::Type desiredType )
    {
        if ( value.getType() != desiredType )
        {
            mlir::Type valType = value.getType();

            if ( valType.isF64() )
            {
                if ( mlir::isa<mlir::IntegerType>( desiredType ) )
                {
                    value = builder.create<mlir::arith::FPToSIOp>( loc, desiredType, value );
                }
                else if ( desiredType.isF32() )
                {
                    value = builder.create<mlir::LLVM::FPExtOp>( loc, desiredType, value );
                }
            }
            else if ( valType.isF32() )
            {
                if ( mlir::isa<mlir::IntegerType>( desiredType ) )
                {
                    value = builder.create<mlir::arith::FPToSIOp>( loc, desiredType, value );
                }
                else if ( desiredType.isF64() )
                {
                    value = builder.create<mlir::LLVM::FPExtOp>( loc, desiredType, value );
                }
            }
            else if ( mlir::IntegerType viType = mlir::cast<mlir::IntegerType>( valType ) )
            {
                unsigned vwidth = viType.getWidth();
                if ( mlir::isa<mlir::FloatType>( desiredType ) )
                {
                    if ( vwidth == 1 )
                    {
                        value = builder.create<mlir::arith::UIToFPOp>( loc, desiredType, value );
                    }
                    else
                    {
                        value = builder.create<mlir::arith::SIToFPOp>( loc, desiredType, value );
                    }
                }
                else if ( mlir::IntegerType miType = mlir::cast<mlir::IntegerType>( desiredType ) )
                {
                    unsigned mwidth = miType.getWidth();
                    if ( ( vwidth == 1 ) && ( mwidth != 1 ) )
                    {
                        // widen bool to integer using unsigned extension:
                        value = builder.create<mlir::arith::ExtUIOp>( loc, desiredType, value );
                    }
                    else if ( vwidth > mwidth )
                    {
                        value = builder.create<mlir::arith::TruncIOp>( loc, desiredType, value );
                    }
                    else if ( vwidth < mwidth )
                    {
                        value = builder.create<mlir::arith::ExtSIOp>( loc, desiredType, value );
                    }
                }
            }
        }

        return value;
    }

    void MLIRListener::enterReturnStatement( SillyParser::ReturnStatementContext *ctx )
    try
    {
        assert( ctx );
        mlir::Location loc = getStartLocation( ctx );

        processReturnLike( loc, ctx->rvalueExpression() );
    }
    CATCH_USER_ERROR

    void MLIRListener::enterExitStatement( SillyParser::ExitStatementContext *ctx )
    try
    {
        assert( ctx );
        mlir::Location loc = getStartLocation( ctx );

        processReturnLike( loc, ctx->rvalueExpression() );
    }
    CATCH_USER_ERROR

    void MLIRListener::processAssignment( mlir::Location loc, SillyParser::RvalueExpressionContext *exprContext,
                                          const std::string &currentVarName, mlir::Value currentIndexExpr )
    {
        silly::DeclareOp declareOp = lookupDeclareForVar( loc, currentVarName );
        mlir::TypeAttr typeAttr = declareOp.getTypeAttr();
        mlir::Type opType = typeAttr.getValue();

        mlir::Value resultValue = parseRvalue( loc, exprContext, opType );

        mlir::SymbolRefAttr symRef = mlir::SymbolRefAttr::get( &dialect.context, currentVarName );
        if ( isa<silly::StringLiteralOp>( resultValue.getDefiningOp() ) )
        {
            mlir::NamedAttribute varNameAttr( builder.getStringAttr( "var_name" ), symRef );

            builder.create<silly::AssignOp>( loc, mlir::TypeRange{}, mlir::ValueRange{ resultValue },
                                             llvm::ArrayRef<mlir::NamedAttribute>{ varNameAttr } );
        }
        else
        {
            if ( currentIndexExpr )
            {
                mlir::Value i = indexTypeCast( loc, currentIndexExpr );

                silly::AssignOp assign = builder.create<silly::AssignOp>( loc, symRef, i, resultValue );

                LLVM_DEBUG( {
                    mlir::OpPrintingFlags flags;
                    flags.enableDebugInfo( true );

                    assign->print( llvm::outs(), flags );
                    llvm::outs() << "\n";
                } );
            }
            else
            {
                mlir::NamedAttribute varNameAttr( builder.getStringAttr( "var_name" ), symRef );

                builder.create<silly::AssignOp>( loc, mlir::TypeRange{}, mlir::ValueRange{ resultValue },
                                                 llvm::ArrayRef<mlir::NamedAttribute>{ varNameAttr } );
            }
        }
    }

    void MLIRListener::enterAssignment( SillyParser::AssignmentContext *ctx )
    try
    {
        assert( ctx );
        mlir::Location loc = getStartLocation( ctx );
        SillyParser::ScalarOrArrayElementContext *lhs = ctx->scalarOrArrayElement();
        assert( lhs );
        assert( lhs->IDENTIFIER() );
        std::string currentVarName = lhs->IDENTIFIER()->getText();

        SillyParser::IndexExpressionContext *indexExpr = lhs->indexExpression();
        mlir::Value currentIndexExpr = mlir::Value{};

        if ( indexExpr )
        {
            currentIndexExpr = parseRvalue( loc, indexExpr->rvalueExpression(), tyI64 );
        }

        processAssignment( loc, ctx->assignmentRvalue()->rvalueExpression(), currentVarName, currentIndexExpr );
    }
    CATCH_USER_ERROR

    mlir::Value MLIRListener::indexTypeCast( mlir::Location loc, mlir::Value val )
    {
        mlir::IndexType indexTy = builder.getIndexType();
        mlir::Type valTy = val.getType();

        if ( valTy == indexTy )
        {
            return val;
        }

        if ( !valTy.isSignlessInteger( 64 ) && valTy.isInteger() )
        {
            val = castOpIfRequired( loc, val, tyI64 );
            valTy = tyI64;
        }

        // Only support i64, or castable to i64, for now
        if ( !valTy.isSignlessInteger( 64 ) )
        {
            // If it's a non-i64 IntegerType, we could cast up to i64, and then cast that to index.
            throw ExceptionWithContext(
                __FILE__, __LINE__, __func__,
                std::format( "{}internal error: NYI: indexTypeCast from type {} is not supported.\n",
                             formatLocation( loc ), mlirTypeToString( valTy ) ) );
        }

        return builder.create<mlir::arith::IndexCastOp>( loc, indexTy, val );
    }

    mlir::Value MLIRListener::parseLogicalOr( mlir::Location loc, SillyParser::ExpressionContext *ctx,
                                              mlir::Type opType )
    {
        SillyParser::ExprLowestContext *expr = dynamic_cast<SillyParser::ExprLowestContext *>( ctx );
        if ( !expr )
        {
            throw ExceptionWithContext(
                __FILE__, __LINE__, __func__,
                std::format( "{}internal error: unexpected ExpressionContext alternative: {}.\n", formatLocation( loc ),
                             ctx->getText() ) );
        }

        SillyParser::BinaryExpressionLowestContext *lowest = expr->binaryExpressionLowest();
        assert( lowest );

        SillyParser::OrExprContext *orCtx = dynamic_cast<SillyParser::OrExprContext *>( lowest );
        if ( !orCtx )
        {
            // No OR: just descend to the next level (AND / single-term)
            return parseBinaryAnd( loc, dynamic_cast<SillyParser::BinaryExpressionAndContext *>( lowest ), opType );
        }

        // We have at least one OR
        std::vector<SillyParser::BinaryExpressionOrContext *> orOperands = orCtx->binaryExpressionOr();

        // First operand (no special case needed)
        mlir::Value result =
            parseBinaryAnd( loc, dynamic_cast<SillyParser::BinaryExpressionAndContext *>( orOperands[0] ), tyI1 );

        for ( size_t i = 1; i < orOperands.size(); ++i )
        {
            mlir::Value rhs =
                parseBinaryAnd( loc, dynamic_cast<SillyParser::BinaryExpressionAndContext *>( orOperands[i] ), tyI1 );

            result = builder.create<silly::OrOp>( loc, tyI1, result, rhs ).getResult();
        }

        if ( opType )
        {
            result = castOpIfRequired( loc, result, opType );
        }

        return result;
    }

    mlir::Value MLIRListener::parseBinaryAnd( mlir::Location loc, SillyParser::BinaryExpressionAndContext *ctx,
                                              mlir::Type opType )
    {
        assert( ctx );

        // Check whether this context actually contains AND operators
        SillyParser::AndExprContext *andCtx = dynamic_cast<SillyParser::AndExprContext *>( ctx );

        if ( !andCtx )
        {
            // No AND operator: descend directly to the next level (equality)
            return parseEquality( loc, dynamic_cast<SillyParser::BinaryExpressionCompareContext *>( ctx ), opType );
        }

        // We have one or more AND operators
        std::vector<SillyParser::BinaryExpressionAndContext *> andOperands = andCtx->binaryExpressionAnd();

        // First operand
        mlir::Value result =
            parseEquality( loc, dynamic_cast<SillyParser::BinaryExpressionCompareContext *>( andOperands[0] ), tyI1 );

        // Fold the remaining ANDs (left associative)
        for ( size_t i = 1; i < andOperands.size(); ++i )
        {
            mlir::Value rhs = parseEquality(
                loc, dynamic_cast<SillyParser::BinaryExpressionCompareContext *>( andOperands[i] ), tyI1 );

            result = builder.create<silly::AndOp>( loc, tyI1, result, rhs ).getResult();
        }

        return castOpIfRequired( loc, result, opType );
    }

    mlir::Value MLIRListener::parseEquality( mlir::Location loc, SillyParser::BinaryExpressionCompareContext *ctx,
                                             mlir::Type opType )
    {
        assert( ctx );
        mlir::Value value{};

        // Check if this is the concrete alternative that has EQ / NE operators
        SillyParser::EqNeExprContext *eqNeCtx = dynamic_cast<SillyParser::EqNeExprContext *>( ctx );

        if ( !eqNeCtx )
        {
            // No EQ or NE operators: descend directly to comparison level
            SillyParser::BinaryExpressionCompareContext *single =
                ctx->getRuleContext<SillyParser::BinaryExpressionCompareContext>( 0 );
            assert( single );
            return parseComparison( loc, single, opType );
        }

        // We have one or more EQ / NE operators
        std::vector<SillyParser::BinaryExpressionCompareContext *> operands = eqNeCtx->binaryExpressionCompare();

        // First (leftmost) operand
        value = parseComparison( loc, operands[0], opType );

        // Fold left-associatively
        for ( size_t i = 1; i < operands.size(); ++i )
        {
            if ( !opType )
            {
                opType = value.getType();
            }

            mlir::Value rhs = parseComparison( loc, operands[i], opType );

            // Determine which operator was used at position (i-1)
            antlr4::tree::TerminalNode *opToken = nullptr;
            std::string opText;

            if ( i - 1 < eqNeCtx->EQUALITY_TOKEN().size() )
            {
                opToken = eqNeCtx->EQUALITY_TOKEN( i - 1 );
                opText = "EQ";
            }
            else if ( i - 1 < eqNeCtx->NOTEQUAL_TOKEN().size() )
            {
                opToken = eqNeCtx->NOTEQUAL_TOKEN( i - 1 );
                opText = "NE";
            }

            if ( !opToken )
            {
                throw ExceptionWithContext( __FILE__, __LINE__, __func__,
                                            std::format( "{}internal error: missing EQ or NE token at index {}\n",
                                                         formatLocation( loc ), i - 1 ) );
            }

            mlir::Value equalityResult;

            if ( opText == "EQ" )
            {
                equalityResult = builder.create<silly::EqualOp>( loc, tyI1, value, rhs ).getResult();
            }
            else if ( opText == "NE" )
            {
                equalityResult = builder.create<silly::NotEqualOp>( loc, tyI1, value, rhs ).getResult();
            }
            else
            {
                throw ExceptionWithContext( __FILE__, __LINE__, __func__,
                                            std::format( "{}internal error: unexpected equality operator: {}\n",
                                                         formatLocation( loc ), opText ) );
            }

            value = equalityResult;
        }

        // Final cast if caller specified a desired type (rare for equality)
        if ( opType )
        {
            value = castOpIfRequired( loc, value, opType );
        }

        return value;
    }

    mlir::Value MLIRListener::parseComparison( mlir::Location loc, SillyParser::BinaryExpressionCompareContext *ctx,
                                               mlir::Type opType )
    {
        assert( ctx );
        mlir::Value value{};

        // Check if this context actually contains comparison operators
        // (the # compareExpr alternative has the repetition)
        SillyParser::CompareExprContext *compareCtx = dynamic_cast<SillyParser::CompareExprContext *>( ctx );

        if ( !compareCtx )
        {
            // No comparison operators : descend directly to additive level
            SillyParser::BinaryExpressionAddSubContext *single =
                ctx->getRuleContext<SillyParser::BinaryExpressionAddSubContext>( 0 );
            assert( single );
            return parseAdditive( loc, single, opType );
        }

        // We have one or more comparison operators
        std::vector<SillyParser::BinaryExpressionAddSubContext *> operands = compareCtx->binaryExpressionAddSub();

        // First (leftmost) operand
        value = parseAdditive( loc, operands[0], opType );

        // Fold left-associatively (though chained comparisons are rare in practice)
        for ( size_t i = 1; i < operands.size(); ++i )
        {
            mlir::Value rhs = parseAdditive( loc, operands[i], opType );

            // Determine which operator was used at position (i-1)
            antlr4::tree::TerminalNode *opToken = nullptr;
            std::string opText;

            if ( i - 1 < compareCtx->LESSTHAN_TOKEN().size() )
            {
                opToken = compareCtx->LESSTHAN_TOKEN( i - 1 );
                opText = "<";
            }
            else if ( i - 1 < compareCtx->GREATERTHAN_TOKEN().size() )
            {
                opToken = compareCtx->GREATERTHAN_TOKEN( i - 1 );
                opText = ">";
            }
            else if ( i - 1 < compareCtx->LESSEQUAL_TOKEN().size() )
            {
                opToken = compareCtx->LESSEQUAL_TOKEN( i - 1 );
                opText = "<=";
            }
            else if ( i - 1 < compareCtx->GREATEREQUAL_TOKEN().size() )
            {
                opToken = compareCtx->GREATEREQUAL_TOKEN( i - 1 );
                opText = ">=";
            }

            if ( !opToken )
            {
                throw ExceptionWithContext( __FILE__, __LINE__, __func__,
                                            std::format( "{}internal error: missing comparison operator at index {}\n",
                                                         formatLocation( loc ), i - 1 ) );
            }

            if ( opText == "<" )
            {
                value = builder.create<silly::LessOp>( loc, tyI1, value, rhs ).getResult();
            }
            else if ( opText == ">" )
            {
                value = builder.create<silly::LessOp>( loc, tyI1, rhs, value ).getResult();
            }
            else if ( opText == "<=" )
            {
                value = builder.create<silly::LessEqualOp>( loc, tyI1, value, rhs ).getResult();
            }
            else if ( opText == ">=" )
            {
                value = builder.create<silly::LessEqualOp>( loc, tyI1, rhs, value ).getResult();
            }
            else
            {
                throw ExceptionWithContext( __FILE__, __LINE__, __func__,
                                            std::format( "{}internal error: unexpected comparison operator: {}\n",
                                                         formatLocation( loc ), opText ) );
            }
        }

        // Final cast (rarely needed for comparisons, but kept for consistency)
        if ( opType )
        {
            value = castOpIfRequired( loc, value, opType );
        }

        return value;
    }

    inline SillyParser::BinaryExpressionMulDivContext *getSingleMulDiv(
        SillyParser::BinaryExpressionAddSubContext *ctx )
    {
        assert( ctx->children.size() == 1 );
        return dynamic_cast<SillyParser::BinaryExpressionMulDivContext *>( ctx->children[0] );
    }

    mlir::Value MLIRListener::parseAdditive( mlir::Location loc, SillyParser::BinaryExpressionAddSubContext *ctx,
                                             mlir::Type opType )
    {
        assert( ctx );
        mlir::Value value{};

        // Check if this context actually contains + or - operators
        // (AddSubExprContext is the alternative that has the repetition)
        SillyParser::AddSubExprContext *addSubCtx = dynamic_cast<SillyParser::AddSubExprContext *>( ctx );

        if ( !addSubCtx )
        {
            // Descend directly to multiplicative level if no +-
            SillyParser::BinaryExpressionMulDivContext *single = getSingleMulDiv( ctx );
            assert( single );
            return parseMultiplicative( loc, single, opType );
        }

        // We have one or more + or - operators
        std::vector<SillyParser::BinaryExpressionMulDivContext *> operands = addSubCtx->binaryExpressionMulDiv();

        // First operand
        value = parseMultiplicative( loc, operands[0], opType );

        // Fold the remaining additions/subtractions left-associatively
        for ( size_t i = 1; i < operands.size(); ++i )
        {
            mlir::Value rhs = parseMultiplicative( loc, operands[i], opType );

            // Determine operator from the token at position (i-1)
            antlr4::tree::TerminalNode *opToken = nullptr;
            if ( i - 1 < addSubCtx->PLUSCHAR_TOKEN().size() )
            {
                opToken = addSubCtx->PLUSCHAR_TOKEN( i - 1 );
            }
            else if ( i - 1 < addSubCtx->MINUS_TOKEN().size() )
            {
                opToken = addSubCtx->MINUS_TOKEN( i - 1 );
            }

            if ( !opToken )
            {
                throw ExceptionWithContext( __FILE__, __LINE__, __func__,
                                            std::format( "{}internal error: missing + or - operator at index {}\n",
                                                         formatLocation( loc ), i - 1 ) );
            }

            std::string opText = opToken->getText();

            if ( !opType )
            {
                opType = biggestTypeOf( value.getType(), rhs.getType() );
            }

            if ( opText == "+" )
            {
                value = builder.create<silly::AddOp>( loc, opType, value, rhs ).getResult();
            }
            else if ( opText == "-" )
            {
                value = builder.create<silly::SubOp>( loc, opType, value, rhs ).getResult();
            }
            else
            {
                throw ExceptionWithContext( __FILE__, __LINE__, __func__,
                                            std::format( "{}internal error: unexpected additive operator: {}\n",
                                                         formatLocation( loc ), opText ) );
            }
        }

        // Final cast if caller (e.g. assignment) specified a desired type
        if ( opType )
        {
            value = castOpIfRequired( loc, value, opType );
        }

        return value;
    }

    inline SillyParser::UnaryExpressionContext *getSingleUnary( SillyParser::BinaryExpressionMulDivContext *ctx )
    {
        assert( ctx->children.size() == 1 );
        return dynamic_cast<SillyParser::UnaryExpressionContext *>( ctx->children[0] );
    }

    mlir::Value MLIRListener::parseMultiplicative( mlir::Location loc, SillyParser::BinaryExpressionMulDivContext *ctx,
                                                   mlir::Type opType )
    {
        assert( ctx );
        mlir::Value value{};

        // Check whether this context actually contains * or / operators
        // (MulDivExprContext is the alternative that has the repetition)
        SillyParser::MulDivExprContext *mulDivCtx = dynamic_cast<SillyParser::MulDivExprContext *>( ctx );

        if ( !mulDivCtx )
        {
            // Descend directly to unary level if no * or /
            SillyParser::UnaryExpressionContext *single = getSingleUnary( ctx );
            assert( single );
            return parseUnary( loc, single, opType );
        }

        // We have one or more * or / operators
        std::vector<SillyParser::UnaryExpressionContext *> operands = mulDivCtx->unaryExpression();

        // First operand
        value = parseUnary( loc, operands[0], opType );

        // Fold the remaining multiplications/divisions left-associatively
        for ( size_t i = 1; i < operands.size(); ++i )
        {
            mlir::Value rhs = parseUnary( loc, operands[i], opType );

            // Determine operator from the token at position (i-1)
            antlr4::tree::TerminalNode *opToken = nullptr;
            if ( ( i - 1 ) < mulDivCtx->TIMES_TOKEN().size() )
            {
                opToken = mulDivCtx->TIMES_TOKEN( i - 1 );
            }
            else if ( ( i - 1 ) < mulDivCtx->DIV_TOKEN().size() )
            {
                opToken = mulDivCtx->DIV_TOKEN( i - 1 );
            }

            if ( !opToken )
            {
                throw ExceptionWithContext( __FILE__, __LINE__, __func__,
                                            std::format( "{}internal error: missing * or / operator at index {}\n",
                                                         formatLocation( loc ), i - 1 ) );
            }

            std::string opText = opToken->getText();

            if ( !opType )
            {
                opType = biggestTypeOf( value.getType(), rhs.getType() );
            }

            if ( opText == "*" )
            {
                value = builder.create<silly::MulOp>( loc, opType, value, rhs ).getResult();
            }
            else if ( opText == "/" )
            {
                value = builder.create<silly::DivOp>( loc, opType, value, rhs ).getResult();
            }
            else
            {
                throw ExceptionWithContext( __FILE__, __LINE__, __func__,
                                            std::format( "{}internal error: unexpected multiplicative operator: {}\n",
                                                         formatLocation( loc ), opText ) );
            }
        }

        if ( opType )
        {
            value = castOpIfRequired( loc, value, opType );
        }

        return value;
    }

    mlir::Value MLIRListener::parseUnary( mlir::Location loc, SillyParser::UnaryExpressionContext *ctx,
                                          mlir::Type opType )
    {
        assert( ctx );
        mlir::Value value{};

        if ( SillyParser::UnaryOpContext *unaryOpCtx = dynamic_cast<SillyParser::UnaryOpContext *>( ctx ) )
        {
            // Case 1: unary operator applied to another unary expression (# unaryOp)
            SillyParser::UnaryOperatorContext *unaryOp = unaryOpCtx->unaryOperator();
            assert( unaryOp );

            // Recurse to the inner unary expression
            value = parseUnary( loc, unaryOpCtx->unaryExpression(), opType );

            std::string opText = unaryOp->getText();

            if ( opText == "-" )
            {
                // Negation
                value = builder.create<silly::NegOp>( loc, value.getType(), value ).getResult();
            }
            else if ( opText == "+" )
            {
                // Unary plus: identity (no-op)
            }
            else if ( opText == "NOT" )
            {
                // NOT x: (x == 0)
                mlir::Value zero = builder.create<mlir::arith::ConstantIntOp>( loc, 0, 64 );
                value = builder.create<silly::EqualOp>( loc, tyI1, value, zero ).getResult();
            }
            else
            {
                throw ExceptionWithContext(
                    __FILE__, __LINE__, __func__,
                    std::format( "{}internal error: unknown unary operator: {}\n", formatLocation( loc ), opText ) );
            }
        }
        else if ( SillyParser::PrimaryContext *primaryCtx = dynamic_cast<SillyParser::PrimaryContext *>( ctx ) )
        {
            // Case 2: no unary operator, just a primary (# primary)
            value = parsePrimary( loc, primaryCtx->primaryExpression(), opType );
        }
        else
        {
            throw ExceptionWithContext(
                __FILE__, __LINE__, __func__,
                std::format( "{}internal error: unknown unary context: {}\n", formatLocation( loc ), ctx->getText() ) );
        }

        assert( value );
        return value;
    }

    mlir::Value MLIRListener::parsePrimary( mlir::Location loc, SillyParser::PrimaryExpressionContext *ctx,
                                            mlir::Type opType )
    {
        assert( ctx );
        mlir::Value value{};

        bool deduceTypeFromOperands = ( opType == mlir::Type{} );

        if ( SillyParser::LitPrimaryContext *litCtx = dynamic_cast<SillyParser::LitPrimaryContext *>( ctx ) )
        {
            // Literal case (# litPrimary)
            SillyParser::LiteralContext *lit = litCtx->literal();
            assert( lit );

            if ( tNode *booleanNode = lit->BOOLEAN_PATTERN() )
            {
                value = parseBoolean( loc, booleanNode->getText() );
            }
            else if ( tNode *integerNode = lit->INTEGER_PATTERN() )
            {
                value = parseInteger( loc, 64, integerNode->getText() );
            }
            else if ( tNode *floatNode = lit->FLOAT_PATTERN() )
            {
                value = parseFloat( loc, tyF64, floatNode->getText() );
            }
            else if ( tNode *stringNode = lit->STRING_PATTERN() )
            {
                std::string s = stripQuotes( loc, stringNode->getText() );

                mlir::StringAttr strAttr = builder.getStringAttr( s );
                silly::StringLiteralOp stringLiteral = builder.create<silly::StringLiteralOp>( loc, tyPtr, strAttr );

                value = stringLiteral.getResult();
            }
            else
            {
                throw ExceptionWithContext( __FILE__, __LINE__, __func__,
                                            std::format( "{}internal error: unknown literal type in primary: {}.\n",
                                                         formatLocation( loc ), ctx->getText() ) );
            }
        }
        else if ( SillyParser::VarPrimaryContext *varCtx = dynamic_cast<SillyParser::VarPrimaryContext *>( ctx ) )
        {
            // Variable / array element (# varPrimary)
            SillyParser::ScalarOrArrayElementContext *scalarOrArrayElement = varCtx->scalarOrArrayElement();
            assert( scalarOrArrayElement );

            tNode *variableNode = scalarOrArrayElement->IDENTIFIER();
            assert( variableNode );
            std::string varName = variableNode->getText();

            silly::DeclareOp declareOp = lookupDeclareForVar( loc, varName );

            mlir::Type varType = declareOp.getTypeAttr().getValue();

            mlir::SymbolRefAttr symRef = mlir::SymbolRefAttr::get( &dialect.context, varName );

            if ( SillyParser::IndexExpressionContext *indexExpr = scalarOrArrayElement->indexExpression() )
            {
                value = parseRvalue( loc, indexExpr->rvalueExpression(), varType );

                mlir::Value i = indexTypeCast( loc, value );

                value = builder.create<silly::LoadOp>( loc, varType, symRef, i );
            }
            else
            {
                if ( deduceTypeFromOperands && declareOp.getSizeAttr() )    // PRINT.
                {
                    if ( mlir::IntegerType ity = mlir::cast<mlir::IntegerType>( varType ) )
                    {
                        unsigned w = ity.getWidth();
                        if ( w == 8 )
                        {
                            varType = tyPtr;
                        }
                    }
                }

                value = builder.create<silly::LoadOp>( loc, varType, symRef, mlir::Value{} );
            }
        }
        else if ( SillyParser::CallPrimaryContext *callCtx = dynamic_cast<SillyParser::CallPrimaryContext *>( ctx ) )
        {
            // Function call (# callPrimary)
            value = handleCall( callCtx->callExpression() );
        }
        else if ( SillyParser::ParenExprContext *parenCtx = dynamic_cast<SillyParser::ParenExprContext *>( ctx ) )
        {
            // Parenthesized expression (# parenExpr)
            value = parseLogicalOr( loc, parenCtx->expression(), opType );
        }
        else
        {
            throw ExceptionWithContext( __FILE__, __LINE__, __func__,
                                        std::format( "{}internal error: unknown primary expression: {}.\n",
                                                     formatLocation( loc ), ctx->getText() ) );
        }

        if ( opType )
        {
            value = castOpIfRequired( loc, value, opType );
        }

        assert( value );
        return value;
    }
}    // namespace silly

// vim: et ts=4 sw=4
