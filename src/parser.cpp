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

    inline void MLIRListener::setVarState( const std::string &funcName, const std::string &varName, VariableState st )
    {
        PerFunctionState &f = funcState( funcName );
        f.varStates[varName] = st;
    }

    inline VariableState MLIRListener::getVarState( const std::string &varName )
    {
        PerFunctionState &f = funcState( currentFuncName );
        return f.varStates[varName];
    }

    inline std::string MLIRListener::formatLocation( mlir::Location loc ) const
    {
        if ( mlir::FileLineColLoc fileLoc = mlir::dyn_cast<mlir::FileLineColLoc>( loc ) )
        {
            return std::format( "{}:{}:{}: ", fileLoc.getFilename().str(), fileLoc.getLine(), fileLoc.getColumn() );
        }
        return "";
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
                                            SillyParser::ArrayBoundsExpressionContext *arrayBounds )
    {
        VariableState varState = getVarState( varName );
        if ( varState != VariableState::undeclared )
        {
            throw UserError( loc, std::format( "Variable {} already declared", varName ) );
        }

        setVarState( currentFuncName, varName, VariableState::declared );

        size_t arraySize{};
        if ( arrayBounds )
        {
            tNode *index = arrayBounds->INTEGER_PATTERN();
            assert( index );
            arraySize = std::stoi( index->getText() );
        }

        mlir::OpBuilder::InsertPoint savedIP = builder.saveInsertionPoint();

        // Get the single scope
        mlir::func::FuncOp funcOp = getFuncOp( loc, currentFuncName );

        silly::ScopeOp scopeOp = getEnclosingScopeOp( loc, funcOp );

        // Scope has one block
        mlir::Block *scopeBlock = &scopeOp.getBody().front();

        // Insert declarations at the beginning of the scope block
        // (all DeclareOps should appear before any scf.if/scf.for)
        builder.setInsertionPointToStart( scopeBlock );

        mlir::StringAttr strAttr = builder.getStringAttr( varName );
        silly::DeclareOp dcl;
        if ( arraySize )
        {
            dcl = builder.create<silly::DeclareOp>( loc, mlir::TypeAttr::get( ty ),
                                                    builder.getI64IntegerAttr( arraySize ),
                                                    /*parameter=*/nullptr, nullptr );
        }
        else
        {
            dcl = builder.create<silly::DeclareOp>( loc, mlir::TypeAttr::get( ty ), nullptr, /*parameter=*/nullptr,
                                                    nullptr );
        }
        dcl->setAttr( "sym_name", strAttr );

        builder.restoreInsertionPoint( savedIP );
    }

    mlir::Value MLIRListener::buildUnaryExpression( mlir::Location loc, tNode *booleanNode, tNode *integerNode,
                                                    tNode *floatNode,
                                                    SillyParser::ScalarOrArrayElementContext *scalarOrArrayElement,
                                                    SillyParser::CallExpressionContext *callNode, tNode *stringNode,
                                                    std::string &s )
    {
        mlir::Value value{};

        if ( callNode )
        {
            value = handleCall( callNode );
        }
        else if ( booleanNode )
        {
            int val;
            std::string bv = booleanNode->getText();
            if ( bv == "TRUE" )
            {
                val = 1;
            }
            else if ( bv == "FALSE" )
            {
                val = 0;
            }
            else
            {
                throw ExceptionWithContext( __FILE__, __LINE__, __func__,
                                            std::format( "{}error: internal error: boolean value neither TRUE nor "
                                                         "FALSE.\n",
                                                         formatLocation( loc ) ) );
            }

            value = builder.create<mlir::arith::ConstantIntOp>( loc, val, 1 );
        }
        else if ( integerNode )
        {
            int64_t val = std::stoll( integerNode->getText() );
            value = builder.create<mlir::arith::ConstantIntOp>( loc, val, 64 );
        }
        else if ( floatNode )
        {
            double val = std::stod( floatNode->getText() );

            llvm::APFloat apVal( val );

            // Like the INTEGER_PATTERN node above, create the float literal with
            // the max sized type. Would need a grammar change to have a
            // specific type (i.e.: size) associated with literals.
            value = builder.create<mlir::arith::ConstantFloatOp>( loc, tyF64, apVal );
        }
        else if ( scalarOrArrayElement )
        {
            tNode *variableNode = scalarOrArrayElement->IDENTIFIER();
            assert( variableNode );
            std::string varName = variableNode->getText();

            VariableState varState = getVarState( varName );

            if ( varState == VariableState::undeclared )
            {
                throw UserError( loc, std::format( "Variable {} not declared in expr", varName ) );
            }

            if ( varState != VariableState::assigned )
            {
                throw UserError( loc, std::format( "Variable {} not assigned in expr", varName ) );
            }

            silly::DeclareOp declareOp = lookupDeclareForVar( loc, varName );

            mlir::Type varType = declareOp.getTypeAttr().getValue();
            mlir::SymbolRefAttr symRef = mlir::SymbolRefAttr::get( &dialect.context, varName );

            mlir::Value indexValue = mlir::Value();

            if ( SillyParser::IndexExpressionContext *indexExpr = scalarOrArrayElement->indexExpression() )
            {
                indexValue = buildNonStringUnaryExpression( loc, nullptr, indexExpr->INTEGER_PATTERN(), nullptr,
                                                            scalarOrArrayElement, nullptr, nullptr );

                mlir::Value i = indexTypeCast( loc, indexValue );

                value = builder.create<silly::LoadOp>( loc, varType, symRef, i );
            }
            else
            {
                value = builder.create<silly::LoadOp>( loc, varType, symRef, mlir::Value() );
            }
        }
        else if ( stringNode )
        {
            s = stripQuotes( loc, stringNode->getText() );
        }
        else
        {
            throw ExceptionWithContext( __FILE__, __LINE__, __func__,
                                        std::format( "{}internal error: Invalid operand\n", formatLocation( loc ) ) );
        }

        return value;
    }

    mlir::Value MLIRListener::buildNonStringUnaryExpression(
        mlir::Location loc, tNode *booleanNode, tNode *integerNode, tNode *floatNode,
        SillyParser::ScalarOrArrayElementContext *scalarOrArrayElement, SillyParser::CallExpressionContext *callNode,
        tNode *stringNode )
    {
        mlir::Value value{};
        std::string s;
        value = buildUnaryExpression( loc, booleanNode, integerNode, floatNode, scalarOrArrayElement, callNode,
                                      stringNode, s );
        if ( s.length() )
        {
            throw ExceptionWithContext(
                __FILE__, __LINE__, __func__,
                std::format( "{}internal error: Unexpected string literal\n", formatLocation( loc ) ) );
        }

        return value;
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

    silly::ScopeOp MLIRListener::getEnclosingScopeOp( mlir::Location loc, mlir::func::FuncOp funcOp ) const
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
            std::format( "{}internal error: Unable to find Enclosing ScopeOp for currentFunction {}\n",
                         formatLocation( loc ), currentFuncName ) );

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
            silly::DeclareOp dcl =
                builder.create<silly::DeclareOp>( startLoc, mlir::TypeAttr::get( argType ), /*size=*/nullptr,
                                                  builder.getUnitAttr(), builder.getI64IntegerAttr( i ) );
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

    void MLIRListener::exitStartRule( SillyParser::StartRuleContext *ctx )
    try
    {
        assert( ctx );
        mlir::Location loc = getStartLocation( ctx );

        if ( !ctx->exitStatement() )
        {
            processReturnLike<SillyParser::NumericLiteralContext>( loc, nullptr, nullptr, nullptr );
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

            setVarState( funcName, paramName, VariableState::assigned );
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
                std::string paramText = p->getText();
                std::cout << std::format( "CALL function {}: param: {}\n", funcName, paramText );

                bool foundStringLiteral{};
                std::string s;
                mlir::Type ty = funcType.getInputs()[i];
                mlir::Value value = parseRvalue( loc, p, ty, s, foundStringLiteral );
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

        registerDeclaration( loc, varName, tyF64, ctx->arrayBoundsExpression() );
    }
    CATCH_USER_ERROR

    void MLIRListener::enterBoolDeclare( SillyParser::BoolDeclareContext *ctx )
    try
    {
        assert( ctx );
        mlir::Location loc = getStartLocation( ctx );
        std::string varName = ctx->IDENTIFIER()->getText();
        registerDeclaration( loc, varName, tyI1, ctx->arrayBoundsExpression() );
    }
    CATCH_USER_ERROR

    void MLIRListener::enterIntDeclare( SillyParser::IntDeclareContext *ctx )
    try
    {
        assert( ctx );
        mlir::Location loc = getStartLocation( ctx );
        assert( ctx->IDENTIFIER() );
        std::string varName = ctx->IDENTIFIER()->getText();

        if ( ctx->INT8_TOKEN() )
        {
            registerDeclaration( loc, varName, tyI8, ctx->arrayBoundsExpression() );
        }
        else if ( ctx->INT16_TOKEN() )
        {
            registerDeclaration( loc, varName, tyI16, ctx->arrayBoundsExpression() );
        }
        else if ( ctx->INT32_TOKEN() )
        {
            registerDeclaration( loc, varName, tyI32, ctx->arrayBoundsExpression() );
        }
        else if ( ctx->INT64_TOKEN() )
        {
            registerDeclaration( loc, varName, tyI64, ctx->arrayBoundsExpression() );
        }
        else
        {
            throw ExceptionWithContext( __FILE__, __LINE__, __func__,
                                        std::format( "{}internal error: Unsupported signed integer declaration size.\n",
                                                     formatLocation( loc ) ) );
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

        if ( ctx->FLOAT32_TOKEN() )
        {
            registerDeclaration( loc, varName, tyF32, ctx->arrayBoundsExpression() );
        }
        else if ( ctx->FLOAT64_TOKEN() )
        {
            registerDeclaration( loc, varName, tyF64, ctx->arrayBoundsExpression() );
        }
        else
        {
            throw ExceptionWithContext( __FILE__, __LINE__, __func__,
                                        std::format( "{}internal error: Unsupported floating point declaration size.\n",
                                                     formatLocation( loc ) ) );
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

        registerDeclaration( loc, varName, tyI8, arrayBounds );
    }
    CATCH_USER_ERROR

    mlir::Value MLIRListener::parsePredicate( mlir::Location loc, SillyParser::BooleanValueContext *booleanValue )
    {
        // booleanValue
        //   : booleanElement | (binaryElement predicateOperator binaryElement)
        //   ;

        mlir::Value conditionPredicate{};

        assert( booleanValue );
        if ( SillyParser::BooleanElementContext *boolElement = booleanValue->booleanElement() )
        {
            SillyParser::BooleanLiteralContext *lit = boolElement->booleanLiteral();
            conditionPredicate = buildNonStringUnaryExpression( loc, lit ? lit->BOOLEAN_PATTERN() : nullptr,
                                                                lit ? lit->INTEGER_PATTERN() : nullptr, nullptr,
                                                                boolElement->scalarOrArrayElement(), nullptr, nullptr );
        }
        else
        {
            // binaryElement : numericLiteral | unaryOperator? IDENTIFIER

            std::vector<SillyParser::BinaryElementContext *> operands = booleanValue->binaryElement();
            if ( operands.size() == 2 )
            {
                mlir::Value lhsValue;
                mlir::Value rhsValue;
                SillyParser::BinaryElementContext *lhs = operands[0];
                assert( lhs );
                SillyParser::BinaryElementContext *rhs = operands[1];
                assert( rhs );

                SillyParser::NumericLiteralContext *llit = lhs->numericLiteral();
                lhsValue = buildNonStringUnaryExpression( loc, nullptr,    // booleanNode
                                                          llit ? llit->INTEGER_PATTERN() : nullptr,
                                                          llit ? llit->FLOAT_PATTERN() : nullptr,
                                                          lhs->scalarOrArrayElement(), nullptr, nullptr );

                SillyParser::NumericLiteralContext *rlit = rhs->numericLiteral();
                rhsValue = buildNonStringUnaryExpression( loc, nullptr,    // booleanNode
                                                          rlit ? rlit->INTEGER_PATTERN() : nullptr,
                                                          rlit ? rlit->FLOAT_PATTERN() : nullptr,
                                                          lhs->scalarOrArrayElement(), nullptr, nullptr );

                assert( booleanValue );
                SillyParser::PredicateOperatorContext *op = booleanValue->predicateOperator();
                assert( op );
                if ( op->LESSTHAN_TOKEN() )
                {
                    conditionPredicate = builder.create<silly::LessOp>( loc, tyI1, lhsValue, rhsValue ).getResult();
                }
                else if ( op->GREATERTHAN_TOKEN() )
                {
                    conditionPredicate = builder.create<silly::LessOp>( loc, tyI1, rhsValue, lhsValue ).getResult();
                }
                else if ( op->LESSEQUAL_TOKEN() )
                {
                    conditionPredicate =
                        builder.create<silly::LessEqualOp>( loc, tyI1, lhsValue, rhsValue ).getResult();
                }
                else if ( op->GREATEREQUAL_TOKEN() )
                {
                    conditionPredicate =
                        builder.create<silly::LessEqualOp>( loc, tyI1, rhsValue, lhsValue ).getResult();
                }
                else if ( op->EQUALITY_TOKEN() )
                {
                    conditionPredicate = builder.create<silly::EqualOp>( loc, tyI1, lhsValue, rhsValue ).getResult();
                }
                else if ( op->NOTEQUAL_TOKEN() )
                {
                    conditionPredicate = builder.create<silly::NotEqualOp>( loc, tyI1, lhsValue, rhsValue ).getResult();
                }
                else
                {
                    throw ExceptionWithContext(
                        __FILE__, __LINE__, __func__,
                        std::format( "{}internal error: Unsupported binary operator in if condition.\n",
                                     formatLocation( loc ) ) );
                }
            }
            else
            {
                throw ExceptionWithContext(
                    __FILE__, __LINE__, __func__,
                    std::format( "{}internal error: Only binary operators supported in if condition (for now).\n",
                                 formatLocation( loc ) ) );
            }
        }

        LLVM_DEBUG( {
            llvm::errs() << "Predicate:\n";
            conditionPredicate.dump();
        } );

        // Ensure the predicate has type i1
        if ( !conditionPredicate.getType().isInteger( 1 ) )
        {
            throw ExceptionWithContext(
                __FILE__, __LINE__, __func__,
                std::format( "{}internal error: if condition must be i1\n", formatLocation( loc ) ) );
        }

        return conditionPredicate;
    }

    void MLIRListener::createIf( mlir::Location loc, SillyParser::BooleanValueContext *booleanValue, bool saveIP )
    {
        mlir::Value conditionPredicate = parsePredicate( loc, booleanValue );

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
        mlir::scf::IfOp ifOp;

        // Temporarily restore the insertion point to right after the scf.if, to search for our current IfOp
        builder.restoreInsertionPoint( insertionPointStack.back() );

        // Now find the scf.if op that is just before the current insertion point
        mlir::Block *currentBlock = builder.getInsertionBlock();
        assert( currentBlock );
        mlir::Block::iterator ip = builder.getInsertionPoint();

        // The insertion point is at the position where new ops would be inserted.
        // So the operation just before it should be the scf.if
        if ( ip != currentBlock->begin() )
        {
            mlir::Operation *prevOp = &*( --ip );    // the op immediately before the insertion point
            ifOp = dyn_cast<mlir::scf::IfOp>( prevOp );
        }

        if ( !ifOp )
        {
            throw ExceptionWithContext(
                __FILE__, __LINE__, __func__,
                std::format( "{}internal error: Could not find scf.if op corresponding to this if statement\n",
                             formatLocation( loc ), errorText ) );
        }

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
        VariableState varState = getVarState( varName );
        if ( varState == VariableState::undeclared )
        {
            throw UserError( loc, std::format( "Variable {} not declared in PRINT", varName ) );
        }

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
        int sl{};
        if ( pStart )
        {
            bool foundStringLiteral{};
            start = parseRvalue( loc, pStart, elemType, s, foundStringLiteral );
            sl += ( foundStringLiteral == true );

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
            bool foundStringLiteral{};
            end = parseRvalue( loc, pEnd, elemType, s, foundStringLiteral );
            sl += ( foundStringLiteral == true );

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
            bool foundStringLiteral{};
            step = parseRvalue( loc, pStep, elemType, s, foundStringLiteral );
            sl += ( foundStringLiteral == true );
        }
        else
        {
            //'scf.for' op failed to verify that all of {lowerBound, upperBound, step} have same type
            step = builder.create<mlir::arith::ConstantIntOp>( loc, 1, 64 );
        }

        step = castOpIfRequired( loc, step, elemType );

        if ( sl )
        {
            throw ExceptionWithContext(
                __FILE__, __LINE__, __func__,
                std::format( "{}internal error: unexpected string literal found while parsing for range: {}\n",
                             formatLocation( loc ), ctx->getText() ) );
        }

        insertionPointStack.push_back( builder.saveInsertionPoint() );

        mlir::scf::ForOp forOp = builder.create<mlir::scf::ForOp>( loc, start, end, step );

        mlir::Block &loopBody = forOp.getRegion().front();
        builder.setInsertionPointToStart( &loopBody );

        // emit an assignment to the variable as the first statement in the loop body, so that any existing references
        // to that will work as-is:
        mlir::Value inductionVar = loopBody.getArgument( 0 );
        builder.create<silly::AssignOp>( loc, mlir::TypeRange{}, mlir::ValueRange{ inductionVar },
                                         llvm::ArrayRef<mlir::NamedAttribute>{ varNameAttr } );
        setVarState( currentFuncName, varName, VariableState::assigned );
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

    void MLIRListener::enterPrint( SillyParser::PrintContext *ctx )
    try
    {
        assert( ctx );
        mlir::Location loc = getStartLocation( ctx );

        mlir::Type varType;

        std::vector<SillyParser::PrintArgumentContext *> args = ctx->printArgument();
        std::vector<mlir::Value> vargs;
        for ( SillyParser::PrintArgumentContext *parg : args )
        {
            mlir::Value v;

            SillyParser::ScalarOrArrayElementContext *scalarOrArrayElement = parg->scalarOrArrayElement();
            if ( scalarOrArrayElement )
            {
                tNode *varNameObject = scalarOrArrayElement->IDENTIFIER();
                assert( varNameObject );
                std::string varName = varNameObject->getText();
                VariableState varState = getVarState( varName );
                if ( varState == VariableState::undeclared )
                {
                    throw UserError( loc, std::format( "Variable {} not declared in PRINT", varName ) );
                }
                if ( varState != VariableState::assigned )
                {
                    throw UserError( loc, std::format( "Variable {} not assigned in PRINT", varName ) );
                }

                silly::DeclareOp declareOp = lookupDeclareForVar( loc, varName );

                mlir::Type elemType = declareOp.getTypeAttr().getValue();
                mlir::Value optIndexValue{};
                if ( SillyParser::IndexExpressionContext *indexExpr = scalarOrArrayElement->indexExpression() )
                {
                    mlir::Value indexValue{};
                    indexValue = buildNonStringUnaryExpression( loc, nullptr, indexExpr->INTEGER_PATTERN(), nullptr,
                                                                scalarOrArrayElement, nullptr, nullptr );

                    optIndexValue = indexTypeCast( loc, indexValue );
                    varType = elemType;
                }
                else if ( declareOp.getSizeAttr() )    // Check if size attribute exists
                {
                    // Array: load a generic pointer (print a string literal)
                    varType = tyPtr;
                }
                else
                {
                    // Scalar: load the value
                    varType = elemType;
                }

                mlir::SymbolRefAttr symRef = mlir::SymbolRefAttr::get( &dialect.context, varName );
                silly::LoadOp value = builder.create<silly::LoadOp>( loc, varType, symRef, optIndexValue );
                v = value.getResult();
            }
            else if ( tNode *theString = parg->STRING_PATTERN() )
            {
                assert( theString );
                std::string s = stripQuotes( loc, theString->getText() );
                mlir::StringAttr strAttr = builder.getStringAttr( s );

                silly::StringLiteralOp stringLiteral = builder.create<silly::StringLiteralOp>( loc, tyPtr, strAttr );
                v = stringLiteral.getResult();
            }
            else if ( SillyParser::NumericLiteralContext *theNumber = parg->numericLiteral() )
            {
                v = buildNonStringUnaryExpression( loc, nullptr, theNumber->INTEGER_PATTERN(),
                                                   theNumber->FLOAT_PATTERN(), nullptr, nullptr, nullptr );
            }
            else if ( SillyParser::BooleanLiteralContext *theBoolean = parg->booleanLiteral() )
            {
                v = buildNonStringUnaryExpression( loc, theBoolean->BOOLEAN_PATTERN(), nullptr, nullptr, nullptr,
                                                   nullptr, nullptr );
            }
            else
            {
                throw ExceptionWithContext( __FILE__, __LINE__, __func__,
                                            std::format( "{}internal error: unexpected print context {}\n",
                                                         formatLocation( loc ), ctx->getText() ) );
            }

            vargs.push_back( v );
        }

        builder.create<silly::PrintOp>( loc, vargs );
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
            VariableState varState = getVarState( varName );
            if ( varState == VariableState::undeclared )
            {
                throw UserError( loc, std::format( "Variable {} not declared in GET", varName ) );
            }

            silly::DeclareOp declareOp = lookupDeclareForVar( loc, varName );

            mlir::Type elemType = declareOp.getTypeAttr().getValue();
            mlir::Value optIndexValue{};
            if ( SillyParser::IndexExpressionContext *indexExpr = scalarOrArrayElement->indexExpression() )
            {
                mlir::Value indexValue{};
                indexValue = buildNonStringUnaryExpression( loc, nullptr, indexExpr->INTEGER_PATTERN(), nullptr,
                                                            scalarOrArrayElement, nullptr, nullptr );

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

            setVarState( currentFuncName, varName, VariableState::assigned );
        }
        else
        {
            throw ExceptionWithContext(
                __FILE__, __LINE__, __func__,
                std::format( "{}internal error: unexpected get context {}\n", formatLocation( loc ), ctx->getText() ) );
        }
    }
    CATCH_USER_ERROR

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

    template <class Literal>
    void MLIRListener::processReturnLike( mlir::Location loc, Literal *lit,
                                          SillyParser::ScalarOrArrayElementContext *scalarOrArrayElement,
                                          tNode *boolNode )
    {
        // Handle the dummy ReturnOp originally inserted in the FuncOp's block
        mlir::Block *currentBlock = builder.getInsertionBlock();

        mlir::Type returnType{};
        bool returnTypeNotEmpty{};

        if ( currentBlock && !currentBlock->empty() )
        {
            mlir::Operation *parentOp = currentBlock->getParentOp();
            assert( parentOp );
            if ( !isa<silly::ScopeOp>( parentOp ) )
            {
                LLVM_DEBUG( {
                    llvm::errs() << std::format( "IP stacking error:\n" );
                    mod.dump();
                } );
                throw ExceptionWithContext(
                    __FILE__, __LINE__, __func__,
                    std::format( "{}internal error: RETURN statement must be inside a silly.scope\n",
                                 formatLocation( loc ) ) );
            }

            mlir::func::FuncOp func = parentOp->getParentOfType<mlir::func::FuncOp>();
            llvm::ArrayRef<mlir::Type> returnTypeArray = func.getFunctionType().getResults();

            if ( !returnTypeArray.empty() )
            {
                returnType = returnTypeArray[0];
                returnTypeNotEmpty = true;
            }
        }
        else
        {
            returnType = tyI32;
            returnTypeNotEmpty = true;
        }

        mlir::Value value{};

        // always regenerate the RETURN/EXIT so that we have the terminator location set properly (not the function body
        // start location that was used to create the dummy silly.ReturnOp that we rewrite here.)
        if ( lit || scalarOrArrayElement || boolNode )
        {
            value = buildNonStringUnaryExpression( loc, boolNode, lit ? lit->INTEGER_PATTERN() : nullptr,
                                                   lit ? lit->FLOAT_PATTERN() : nullptr, scalarOrArrayElement, nullptr,
                                                   nullptr );

            // Apply type conversions to match func::FuncOp return type.  This is adapted from AssignOpLowering, but
            // uses arith dialect operations instead of LLVM dialect
            if ( returnTypeNotEmpty )
            {
                value = castOpIfRequired( loc, value, returnType );
            }
        }
        else
        {
            if ( returnTypeNotEmpty )
            {
                if ( mlir::IntegerType intType = mlir::dyn_cast<mlir::IntegerType>( returnType ) )
                {
                    unsigned width = intType.getWidth();
                    value = builder.create<mlir::arith::ConstantIntOp>( loc, 0, width );
                }
                else if ( mlir::FloatType floatType = mlir::dyn_cast<mlir::FloatType>( returnType ) )
                {
                    llvm::APFloat apVal( 0.0 );
                    unsigned w = floatType.getWidth();
                    if ( w == 64 )
                    {
                        value = builder.create<mlir::arith::ConstantFloatOp>( loc, tyF64, apVal );
                    }
                    else if ( w == 32 )
                    {
                        value = builder.create<mlir::arith::ConstantFloatOp>( loc, tyF32, apVal );
                    }
                    else
                    {
                        throw ExceptionWithContext( __FILE__, __LINE__, __func__,
                                                    std::format( "{}internal error: Support for FloatType w/ size "
                                                                 "other than 32/64 is not implemented: {}\n",
                                                                 formatLocation( loc ), w ) );
                    }
                }
                else
                {
                    throw ExceptionWithContext(
                        __FILE__, __LINE__, __func__,
                        std::format(
                            "{}internal error: Return support for non-(IntegerType,FloatType) is not implemented\n",
                            formatLocation( loc ) ) );
                }
            }
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

    void MLIRListener::enterReturnStatement( SillyParser::ReturnStatementContext *ctx )
    try
    {
        assert( ctx );
        mlir::Location loc = getStartLocation( ctx );

        SillyParser::LiteralContext *lit = ctx->literal();
        processReturnLike<SillyParser::LiteralContext>( loc, lit, ctx->scalarOrArrayElement(),
                                                        lit ? lit->BOOLEAN_PATTERN() : nullptr );
    }
    CATCH_USER_ERROR

    void MLIRListener::enterExitStatement( SillyParser::ExitStatementContext *ctx )
    try
    {
        assert( ctx );
        mlir::Location loc = getStartLocation( ctx );

        SillyParser::NumericLiteralContext *lit = ctx->numericLiteral();

        processReturnLike<SillyParser::NumericLiteralContext>( loc, lit, ctx->scalarOrArrayElement(), nullptr );
    }
    CATCH_USER_ERROR

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
        mlir::Value currentIndexExpr = mlir::Value();

        if ( indexExpr )
        {
            currentIndexExpr = buildNonStringUnaryExpression( loc, nullptr, indexExpr->INTEGER_PATTERN(), nullptr, lhs,
                                                              nullptr, nullptr );
        }

        VariableState varState = getVarState( currentVarName );
        if ( varState == VariableState::declared )
        {
            setVarState( currentFuncName, currentVarName, VariableState::assigned );
        }
        else if ( varState == VariableState::undeclared )
        {
            throw UserError( loc, std::format( "Variable {} not declared in assignment", currentVarName ) );
        }

        assert( ctx->assignmentRvalue() );
        SillyParser::RvalueExpressionContext *exprContext = ctx->assignmentRvalue()->rvalueExpression();

        silly::DeclareOp declareOp = lookupDeclareForVar( loc, currentVarName );
        mlir::TypeAttr typeAttr = declareOp.getTypeAttr();
        mlir::Type opType = typeAttr.getValue();

        std::string s;
        bool foundStringLiteral{};
        mlir::Value resultValue = parseRvalue( loc, exprContext, opType, s, foundStringLiteral );

        mlir::SymbolRefAttr symRef = mlir::SymbolRefAttr::get( &dialect.context, currentVarName );
        if ( foundStringLiteral )
        {
            mlir::StringAttr strAttr = builder.getStringAttr( s );

            silly::StringLiteralOp stringLiteral = builder.create<silly::StringLiteralOp>( loc, tyPtr, strAttr );

            mlir::NamedAttribute varNameAttr( builder.getStringAttr( "var_name" ), symRef );

            builder.create<silly::AssignOp>( loc, mlir::TypeRange{}, mlir::ValueRange{ stringLiteral },
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

        setVarState( currentFuncName, currentVarName, VariableState::assigned );
    }
    CATCH_USER_ERROR

    mlir::Value MLIRListener::indexTypeCast( mlir::Location loc, mlir::Value val )
    {
        mlir::IndexType indexTy = builder.getIndexType();
        mlir::Type valTy = val.getType();

        if ( valTy == indexTy )
            return val;

        // Only support i64 for now
        if ( !valTy.isSignlessInteger( 64 ) )
        {
            // If it's a non-i64 IntegerType, we could cast up to i64, and then cast that to index.
            throw ExceptionWithContext( __FILE__, __LINE__, __func__,
                                        std::format( "{}internal error: NYI: indexTypeCast for types other than i64\n",
                                                     formatLocation( loc ) ) );
        }

        return builder.create<mlir::arith::IndexCastOp>( loc, indexTy, val );
    }

    mlir::Value MLIRListener::parseRvalue( mlir::Location loc, SillyParser::RvalueExpressionContext *ctx,
                                           mlir::Type opType, std::string &s, bool &foundStringLiteral )
    {
        mlir::Value resultValue;
        mlir::Value lhsValue;
        size_t bsz = ctx->binaryElement().size();

        if ( bsz == 0 )
        {
            mlir::Value lhsValue;

            SillyParser::LiteralContext *lit = ctx->literal();
            lhsValue = buildUnaryExpression( loc, lit ? lit->BOOLEAN_PATTERN() : nullptr,
                                             lit ? lit->INTEGER_PATTERN() : nullptr,
                                             lit ? lit->FLOAT_PATTERN() : nullptr, ctx->scalarOrArrayElement(),
                                             ctx->callExpression(), lit ? lit->STRING_PATTERN() : nullptr, s );

            if ( lit && lit->STRING_PATTERN() )
            {
                foundStringLiteral = true;
            }

            resultValue = lhsValue;
            if ( SillyParser::UnaryOperatorContext *unaryOp = ctx->unaryOperator() )
            {
                std::string opText = unaryOp->getText();
                if ( opText == "-" )
                {
                    silly::NegOp op = builder.create<silly::NegOp>( loc, opType, lhsValue );
                    resultValue = op.getResult();

                    if ( s.length() )
                    {
                        throw ExceptionWithContext(
                            __FILE__, __LINE__, __func__,
                            std::format( "{}internal error: Unexpected string literal for negation operation\n",
                                         formatLocation( loc ) ) );
                    }
                }
                else if ( opText == "NOT" )
                {
                    mlir::arith::ConstantIntOp rhsValue = builder.create<mlir::arith::ConstantIntOp>( loc, 0, 64 );

                    silly::EqualOp b = builder.create<silly::EqualOp>( loc, opType, lhsValue, rhsValue );
                    resultValue = b.getResult();

                    if ( s.length() )
                    {
                        throw ExceptionWithContext(
                            __FILE__, __LINE__, __func__,
                            std::format( "{}internal error: Unexpected string literal for negation operation\n",
                                         formatLocation( loc ) ) );
                    }
                }
            }
        }
        else
        {
            if ( bsz != 2 )
            {
                throw ExceptionWithContext(
                    __FILE__, __LINE__, __func__,
                    std::format( "{}internal error: binaryElement size != 2\n", formatLocation( loc ) ) );
            }

            SillyParser::BinaryElementContext *lhs = ctx->binaryElement()[0];
            assert( lhs );
            SillyParser::BinaryElementContext *rhs = ctx->binaryElement()[1];
            assert( rhs );
            std::string opText = ctx->binaryOperator()->getText();

            SillyParser::NumericLiteralContext *llit = lhs->numericLiteral();
            lhsValue = buildNonStringUnaryExpression( loc, nullptr,    // booleanNode
                                                      llit ? llit->INTEGER_PATTERN() : nullptr,
                                                      llit ? llit->FLOAT_PATTERN() : nullptr,
                                                      lhs->scalarOrArrayElement(), lhs->callExpression(), nullptr );

            mlir::Value rhsValue;
            SillyParser::NumericLiteralContext *rlit = rhs->numericLiteral();
            rhsValue = buildNonStringUnaryExpression( loc, nullptr,    // booleanNode
                                                      rlit ? rlit->INTEGER_PATTERN() : nullptr,
                                                      rlit ? rlit->FLOAT_PATTERN() : nullptr,
                                                      rhs->scalarOrArrayElement(), rhs->callExpression(), nullptr );

            // Create the binary operator (supports +, -, *, /)
            if ( opText == "+" )
            {
                silly::AddOp b = builder.create<silly::AddOp>( loc, opType, lhsValue, rhsValue );
                resultValue = b.getResult();
            }
            else if ( opText == "-" )
            {
                silly::SubOp b = builder.create<silly::SubOp>( loc, opType, lhsValue, rhsValue );
                resultValue = b.getResult();
            }
            else if ( opText == "*" )
            {
                silly::MulOp b = builder.create<silly::MulOp>( loc, opType, lhsValue, rhsValue );
                resultValue = b.getResult();
            }
            else if ( opText == "/" )
            {
                silly::DivOp b = builder.create<silly::DivOp>( loc, opType, lhsValue, rhsValue );
                resultValue = b.getResult();
            }
            else if ( opText == "<" )
            {
                silly::LessOp b = builder.create<silly::LessOp>( loc, opType, lhsValue, rhsValue );
                resultValue = b.getResult();
            }
            else if ( opText == ">" )
            {
                silly::LessOp b = builder.create<silly::LessOp>( loc, opType, rhsValue, lhsValue );
                resultValue = b.getResult();
            }
            else if ( opText == "<=" )
            {
                silly::LessEqualOp b = builder.create<silly::LessEqualOp>( loc, opType, lhsValue, rhsValue );
                resultValue = b.getResult();
            }
            else if ( opText == ">=" )
            {
                silly::LessEqualOp b = builder.create<silly::LessEqualOp>( loc, opType, rhsValue, lhsValue );
                resultValue = b.getResult();
            }
            else if ( opText == "EQ" )
            {
                silly::EqualOp b = builder.create<silly::EqualOp>( loc, opType, lhsValue, rhsValue );
                resultValue = b.getResult();
            }
            else if ( opText == "NE" )
            {
                silly::NotEqualOp b = builder.create<silly::NotEqualOp>( loc, opType, lhsValue, rhsValue );
                resultValue = b.getResult();
            }
            else if ( opText == "AND" )
            {
                silly::AndOp b = builder.create<silly::AndOp>( loc, opType, lhsValue, rhsValue );
                resultValue = b.getResult();
            }
            else if ( opText == "OR" )
            {
                silly::OrOp b = builder.create<silly::OrOp>( loc, opType, lhsValue, rhsValue );
                resultValue = b.getResult();
            }
            else if ( opText == "XOR" )
            {
                silly::XorOp b = builder.create<silly::XorOp>( loc, opType, lhsValue, rhsValue );
                resultValue = b.getResult();
            }
            else
            {
                throw ExceptionWithContext(
                    __FILE__, __LINE__, __func__,
                    std::format( "{}internal error: Invalid binary operator {}\n", formatLocation( loc ), opText ) );
            }
        }

        return resultValue;
    }
}    // namespace silly

// vim: et ts=4 sw=4
