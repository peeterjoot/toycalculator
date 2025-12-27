/**
 * @file    parser.cpp
 * @author  Peeter Joot <peeterjoot@pm.me>
 * @brief   altlr4 parse tree listener and MLIR builder.
 */
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

#include "ToyExceptions.hpp"
#include "constants.hpp"
#include "parser.hpp"

#define DEBUG_TYPE "toy-parser"

namespace toy
{
    inline PerFunctionState &MLIRListener::funcState( const std::string &funcName )
    {
        if ( !pr_funcState.contains( funcName ) )
        {
            pr_funcState[funcName] = std::make_unique<PerFunctionState>( builder.getUnknownLoc() );
        }

        return *pr_funcState[funcName];
    }

    inline void MLIRListener::setVarState( const std::string &funcName, const std::string &varName, variable_state st )
    {
        PerFunctionState &f = funcState( funcName );
        f.varStates[varName] = st;
    }

    inline variable_state MLIRListener::getVarState( const std::string &varName )
    {
        PerFunctionState &f = funcState( currentFuncName );
        return f.varStates[varName];
    }

    inline void MLIRListener::setFuncOp( mlir::Operation *op )
    {
        PerFunctionState &f = funcState( currentFuncName );
        f.funcOp = op;
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
            throw exception_with_context(
                __FILE__, __LINE__, __func__,
                std::format( "{}internal error: String '{}' was not double quotes enclosed as expected.\n",
                             formatLocation( loc ), input ) );
        }

        return input.substr( 1, input.size() - 2 );
    }

    inline mlir::func::FuncOp MLIRListener::getFuncOp( mlir::Location loc, const std::string &funcName )
    {
        PerFunctionState &f = funcState( funcName );
        mlir::func::FuncOp op = mlir::cast<mlir::func::FuncOp>( f.funcOp );

        if ( op == nullptr )
        {
            throw exception_with_context(
                __FILE__, __LINE__, __func__,
                std::format( "{}error: Unable to find FuncOp for function {}\n", formatLocation( loc ), funcName ) );
        }
        return op;
    }

    inline void MLIRListener::markExplicitTerminator()
    {
        PerFunctionState &f = funcState( currentFuncName );
        f.terminatorWasExplcit = true;
    }

    inline bool MLIRListener::wasTerminatorExplicit()
    {
        PerFunctionState &f = funcState( currentFuncName );
        return f.terminatorWasExplcit;
    }

    inline void MLIRListener::setLastLoc( mlir::Location loc )
    {
        PerFunctionState &f = funcState( currentFuncName );
        f.lastLoc = loc;
    }

    inline mlir::Location MLIRListener::getLastLoc()
    {
        PerFunctionState &f = funcState( currentFuncName );
        return f.lastLoc;
    }

    inline mlir::Location MLIRListener::getLocation( antlr4::ParserRuleContext *ctx, bool useStopLocation )
    {
        size_t line = 1;
        size_t col = 0;
        if ( ctx )
        {
            antlr4::Token *tok = useStopLocation ? ctx->getStop() : ctx->getStart();
            assert( tok );
            line = tok->getLine();
            col = tok->getCharPositionInLine();
        }

        mlir::FileLineColLoc loc = mlir::FileLineColLoc::get( builder.getStringAttr( filename ), line, col + 1 );
        lastLocation = loc;

        return loc;
    }

    DialectCtx::DialectCtx()
    {
        context.getOrLoadDialect<toy::ToyDialect>();
        context.getOrLoadDialect<mlir::func::FuncDialect>();
        context.getOrLoadDialect<mlir::arith::ArithDialect>();
        context.getOrLoadDialect<mlir::memref::MemRefDialect>();
        context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
        context.getOrLoadDialect<mlir::scf::SCFDialect>();
    }

    void MLIRListener::registerDeclaration( mlir::Location loc, const std::string &varName, mlir::Type ty,
                                            ToyParser::ArrayBoundsExpressionContext *arrayBounds )
    {
        variable_state varState = getVarState( varName );
        if ( varState != variable_state::undeclared )
        {
            throw exception_with_context(
                __FILE__, __LINE__, __func__,
                std::format( "{}error: Variable {} already declared\n", formatLocation( loc ), varName ) );
        }

        setVarState( currentFuncName, varName, variable_state::declared );

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

        toy::ScopeOp scopeOp = getEnclosingScopeOp( loc, funcOp );

        // Scope has one block
        mlir::Block *scopeBlock = &scopeOp.getBody().front();

        // Insert declarations at the beginning of the scope block
        // (all DeclareOps should appear before any scf.if/scf.for)
        builder.setInsertionPointToStart( scopeBlock );

        mlir::StringAttr strAttr = builder.getStringAttr( varName );
        toy::DeclareOp dcl;
        if ( arraySize )
        {
            dcl =
                builder.create<toy::DeclareOp>( loc, mlir::TypeAttr::get( ty ), builder.getI64IntegerAttr( arraySize ),
                                                /*parameter=*/nullptr, nullptr );
        }
        else
        {
            dcl = builder.create<toy::DeclareOp>( loc, mlir::TypeAttr::get( ty ), nullptr, /*parameter=*/nullptr,
                                                  nullptr );
        }
        dcl->setAttr( "sym_name", strAttr );

        builder.restoreInsertionPoint( savedIP );
    }

    mlir::Value MLIRListener::buildUnaryExpression( mlir::Location loc, tNode *booleanNode, tNode *integerNode,
                                                    tNode *floatNode,
                                                    ToyParser::ScalarOrArrayElementContext *scalarOrArrayElement,
                                                    tNode *stringNode, std::string &s )
    {
        mlir::Value value{};

        if ( booleanNode )
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
                throw exception_with_context( __FILE__, __LINE__, __func__,
                                              std::format( "{}error: Internal error: boolean value neither TRUE nor "
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

            variable_state varState = getVarState( varName );

            if ( varState == variable_state::undeclared )
            {
                throw exception_with_context(
                    __FILE__, __LINE__, __func__,
                    std::format( "{}error: Variable {} not declared in expr\n", formatLocation( loc ), varName ) );
            }

            if ( varState != variable_state::assigned )
            {
                throw exception_with_context(
                    __FILE__, __LINE__, __func__,
                    std::format( "{}error: Variable {} not assigned in expr\n", formatLocation( loc ), varName ) );
            }

            toy::DeclareOp declareOp = lookupDeclareForVar( loc, varName );

            mlir::Type varType = declareOp.getTypeAttr().getValue();
            mlir::SymbolRefAttr symRef = mlir::SymbolRefAttr::get( &dialect.context, varName );

            mlir::Value indexValue = mlir::Value();

            if ( ToyParser::IndexExpressionContext *indexExpr = scalarOrArrayElement->indexExpression() )
            {
                indexValue = buildNonStringUnaryExpression( loc, nullptr, indexExpr->INTEGER_PATTERN(), nullptr,
                                                            scalarOrArrayElement, nullptr );

                mlir::Value i = indexTypeCast( loc, indexValue );

                value = builder.create<toy::LoadOp>( loc, varType, symRef, i );
            }
            else
            {
                value = builder.create<toy::LoadOp>( loc, varType, symRef, mlir::Value() );
            }
        }
        else if ( stringNode )
        {
            s = stripQuotes( loc, stringNode->getText() );
        }
        else
        {
            throw exception_with_context( __FILE__, __LINE__, __func__,
                                          std::format( "{}error: Invalid operand\n", formatLocation( loc ) ) );
        }

        return value;
    }

    mlir::Value MLIRListener::buildNonStringUnaryExpression(
        mlir::Location loc, tNode *booleanNode, tNode *integerNode, tNode *floatNode,
        ToyParser::ScalarOrArrayElementContext *scalarOrArrayElement, tNode *stringNode )
    {
        mlir::Value value{};
        std::string s;
        value = buildUnaryExpression( loc, booleanNode, integerNode, floatNode, scalarOrArrayElement, stringNode, s );
        if ( s.length() )
        {
            throw exception_with_context(
                __FILE__, __LINE__, __func__,
                std::format( "{}error: Unexpected string literal\n", formatLocation( loc ) ) );
        }

        return value;
    }

    void MLIRListener::syntaxError( antlr4::Recognizer *recognizer, antlr4::Token *offendingSymbol, size_t line,
                                    size_t charPositionInLine, const std::string &msg, std::exception_ptr e )
    {
        hasErrors = true;
        std::string tokenText = offendingSymbol ? offendingSymbol->getText() : "<none>";
        throw exception_with_context( __FILE__, __LINE__, __func__,
                                      std::format( "Syntax error in {}:{}:{}: {} (token: {} )", filename, line,
                                                   charPositionInLine, msg, tokenText ) );
    }

    toy::ScopeOp MLIRListener::getEnclosingScopeOp( mlir::Location loc, mlir::func::FuncOp funcOp ) const
    {
        // Single ScopeOp per function – iterate once to find it
        for ( mlir::Operation &op : funcOp.getBody().front() )
        {
            if ( toy::ScopeOp scopeOp = dyn_cast<toy::ScopeOp>( &op ) )
            {
                return scopeOp;
            }
        }

        throw exception_with_context( __FILE__, __LINE__, __func__,
                                      std::format( "{}error: Unable to find Enclosing ScopeOp for currentFunction {}\n",
                                                   formatLocation( loc ), currentFuncName ) );

        return nullptr;
    }

    toy::DeclareOp MLIRListener::lookupDeclareForVar( mlir::Location loc, const std::string &varName )
    {
        mlir::func::FuncOp funcOp = getFuncOp( loc, currentFuncName );

        toy::ScopeOp scopeOp = getEnclosingScopeOp( loc, funcOp );

        LLVM_DEBUG( {
            llvm::errs() << std::format( "Lookup symbol {} in parent function:\n", varName );
            scopeOp->dump();
        } );

        mlir::Operation *symbolOp = mlir::SymbolTable::lookupSymbolIn( scopeOp, varName );
        if ( !symbolOp )
        {
            throw exception_with_context(
                __FILE__, __LINE__, __func__,
                std::format( "{}error: Undeclared variable {}", formatLocation( loc ), varName ) );
        }

        toy::DeclareOp declareOp = mlir::dyn_cast<toy::DeclareOp>( symbolOp );
        if ( !declareOp )
        {
            throw exception_with_context(
                __FILE__, __LINE__, __func__,
                std::format( "{}error: Undeclared variable {}", formatLocation( loc ), varName ) );
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

    MLIRListener::MLIRListener( const std::string &_filename )
        : filename( _filename ),
          dialect(),
          builder( &dialect.context ),
          currentAssignLoc( getLocation( nullptr ) ),
          mod( mlir::ModuleOp::create( currentAssignLoc ) )
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

    void MLIRListener::createScope( mlir::Location loc, mlir::func::FuncOp funcOp, const std::string &funcName,
                                    const std::vector<std::string> &paramNames )
    {
        mlir::Block &block = *funcOp.addEntryBlock();
        builder.setInsertionPointToStart( &block );

        // initially with empty operands and results
        toy::ScopeOp scopeOp = builder.create<toy::ScopeOp>( loc, mlir::TypeRange{}, mlir::ValueRange{} );
        builder.create<toy::YieldOp>( loc );

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
            toy::DeclareOp dcl = builder.create<toy::DeclareOp>( loc, mlir::TypeAttr::get( argType ), /*size=*/nullptr,
                                                       builder.getUnitAttr(), builder.getI64IntegerAttr( i ) );
            dcl->setAttr( "sym_name", strAttr );
        }

        // Insert a default toy::ReturnOp terminator (with no operands, or default zero return for scalar return
        // functions, like main). This will be replaced later with an toy::ReturnOp with the actual return code if
        // desired.
        mlir::TypeRange returnType = funcOp.getFunctionType().getResults();
        mlir::Operation *returnOp = nullptr;
        if ( !returnType.empty() )
        {
            mlir::arith::ConstantOp zero = builder.create<mlir::arith::ConstantOp>( loc, returnType[0],
                                                                 builder.getIntegerAttr( returnType[0], 0 ) );
            returnOp = builder.create<toy::ReturnOp>( loc, mlir::ValueRange{ zero } );
        }
        else
        {
            returnOp = builder.create<toy::ReturnOp>( loc, mlir::ValueRange{} );
        }


        LLVM_DEBUG( {
            llvm::errs() << std::format( "Created mlir::func::FuncOp stub for function {}\n", funcName );
            funcOp.dump();
        } );

        builder.setInsertionPoint( returnOp );    // Set insertion point *before* the saved toy::ReturnOp

        currentFuncName = funcName;
        setFuncOp( funcOp );
    }

    void MLIRListener::mainFirstTime( mlir::Location loc )
    {
        if ( !mainScopeGenerated && ( MLIRListener::currentFuncName == ENTRY_SYMBOL_NAME ) )
        {
            mlir::FunctionType funcType = builder.getFunctionType( {}, tyI32 );
            mlir::func::FuncOp funcOp = builder.create<mlir::func::FuncOp>( loc, ENTRY_SYMBOL_NAME, funcType );

            std::vector<std::string> paramNames;
            createScope( loc, funcOp, ENTRY_SYMBOL_NAME, paramNames );

            mainScopeGenerated = true;
        }
    }

    void MLIRListener::enterStartRule( ToyParser::StartRuleContext *ctx )
    {
        assert( ctx );
        currentFuncName = ENTRY_SYMBOL_NAME;
        mlir::Location loc = getLocation( ctx );
        setLastLoc( loc );
    }

    void MLIRListener::exitStartRule( ToyParser::StartRuleContext *ctx )
    {
        assert( ctx );
        mlir::Location loc = getLocation( ctx );
        mainFirstTime( loc );

        mlir::Location lastLoc = getLastLoc();

        // LLVM_DEBUG( { llvm::errs() << "exitStartRule module dump before return rewrite:\n"; mod->dump(); } );
        if ( !wasTerminatorExplicit() )
        {
            processReturnLike<ToyParser::NumericLiteralContext>( lastLoc, nullptr, nullptr, nullptr );
        }

        LLVM_DEBUG( {
            llvm::errs() << "exitStartRule done: module dump:\n";
            mod->dump();
        } );
    }

    void MLIRListener::enterFunction( ToyParser::FunctionContext *ctx )
    {
        assert( ctx );
        mlir::Location loc = getLocation( ctx );

        mainIP = builder.saveInsertionPoint();

        builder.setInsertionPointToStart( mod.getBody() );

        assert( ctx->IDENTIFIER() );
        std::string funcName = ctx->IDENTIFIER()->getText();

        std::vector<mlir::Type> returns;
        if ( ToyParser::ScalarTypeContext * rt = ctx->scalarType() )
        {
            mlir::Type returnType = parseScalarType( rt->getText() );
            returns.push_back( returnType );
        }

        std::vector<mlir::Type> paramTypes;
        std::vector<std::string> paramNames;
        for ( ToyParser::VariableTypeAndNameContext *paramCtx : ctx->variableTypeAndName() )
        {
            assert( paramCtx->scalarType() );
            mlir::Type paramType = parseScalarType( paramCtx->scalarType()->getText() );
            assert( paramCtx->IDENTIFIER() );
            std::string paramName = paramCtx->IDENTIFIER()->getText();
            paramTypes.push_back( paramType );
            paramNames.push_back( paramName );

            setVarState( funcName, paramName, variable_state::assigned );
        }

        std::vector<mlir::NamedAttribute> attrs;
        attrs.push_back(
            mlir::NamedAttribute( builder.getStringAttr( "sym_visibility" ), builder.getStringAttr( "private" ) ) );

        mlir::FunctionType funcType = builder.getFunctionType( paramTypes, returns );
        mlir::func::FuncOp funcOp = builder.create<mlir::func::FuncOp>( loc, funcName, funcType, attrs );
        createScope( loc, funcOp, funcName, paramNames );
        setLastLoc( loc );
    }

    void MLIRListener::exitFunction( ToyParser::FunctionContext *ctx )
    {
        assert( ctx );
        mlir::Location lastLoc = getLastLoc();

        // This is in case the grammar enforcement of a RETURN at end of FUNCTION is removed (which would make sense,
        // and is also desirable when control flow is added.) For now, still have enforced mandatory RETURN at
        // function-end in the grammar.
        if ( !wasTerminatorExplicit() )
        {
            processReturnLike<ToyParser::LiteralContext>( lastLoc, nullptr, nullptr, nullptr );
        }

        builder.restoreInsertionPoint( mainIP );

        currentFuncName = ENTRY_SYMBOL_NAME;
    }

    mlir::Value MLIRListener::handleCall( ToyParser::CallContext *ctx )
    {
        assert( ctx );
        mlir::Location loc = getLocation( ctx );
        tNode * id = ctx->IDENTIFIER();
        assert( id );
        std::string funcName = id->getText();
        mlir::func::FuncOp funcOp = getFuncOp( loc, funcName );
        mlir::FunctionType funcType = funcOp.getFunctionType();
        std::vector<mlir::Value> parameters;
        if ( ToyParser::ParameterListContext * params = ctx->parameterList() )
        {
            int i = 0;

            assert( params );
            size_t psz = params->parameter().size();
            size_t fsz = funcType.getInputs().size();
            assert( psz == fsz );

            for ( ToyParser::ParameterContext *p : params->parameter() )
            {
                std::string paramText = p->getText();
                std::cout << std::format( "CALL function {}: param: {}\n", funcName, paramText );

                mlir::Value value;
                ToyParser::LiteralContext * lit = p->literal();

                // Want to support passing string literals (not just to PRINT builtin), but not now.
                value = buildNonStringUnaryExpression( loc, lit ? lit->BOOLEAN_PATTERN() : nullptr,
                                                       lit ? lit->INTEGER_PATTERN() : nullptr,
                                                       lit ? lit->FLOAT_PATTERN() : nullptr, p->scalarOrArrayElement(),
                                                       lit ? lit->STRING_PATTERN() : nullptr );

                value = castOpIfRequired( loc, value, funcType.getInputs()[i] );
                parameters.push_back( value );
                i++;
            }
        }

        mlir::TypeRange resultTypes = funcType.getResults();

        toy::CallOp callOp = builder.create<toy::CallOp>( loc, resultTypes, funcName, parameters );

        // Return the first result (or null for void calls)
        return resultTypes.empty() ? mlir::Value{} : callOp.getResults()[0];
    }

    void MLIRListener::enterCall( ToyParser::CallContext *ctx )
    {
        assert( ctx );
        if ( callIsHandled )
            return;

        mlir::Location loc = getLocation( ctx );
        mainFirstTime( loc );
        setLastLoc( loc );

        handleCall( ctx );
    }

    void MLIRListener::enterDeclare( ToyParser::DeclareContext *ctx )
    {
        assert( ctx );
        mlir::Location loc = getLocation( ctx );
        mainFirstTime( loc );
        setLastLoc( loc );
        assert( ctx->IDENTIFIER() );
        std::string varName = ctx->IDENTIFIER()->getText();

        registerDeclaration( loc, varName, tyF64, ctx->arrayBoundsExpression() );
    }

    void MLIRListener::enterBoolDeclare( ToyParser::BoolDeclareContext *ctx )
    {
        assert( ctx );
        mlir::Location loc = getLocation( ctx );
        mainFirstTime( loc );
        setLastLoc( loc );
        std::string varName = ctx->IDENTIFIER()->getText();
        registerDeclaration( loc, varName, tyI1, ctx->arrayBoundsExpression() );
    }

    void MLIRListener::enterIntDeclare( ToyParser::IntDeclareContext *ctx )
    {
        assert( ctx );
        mlir::Location loc = getLocation( ctx );
        mainFirstTime( loc );
        setLastLoc( loc );
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
            throw exception_with_context(
                __FILE__, __LINE__, __func__,
                std::format( "{}Internal error: Unsupported signed integer declaration size.\n",
                             formatLocation( loc ) ) );
        }
    }

    void MLIRListener::enterFloatDeclare( ToyParser::FloatDeclareContext *ctx )
    {
        assert( ctx );
        mlir::Location loc = getLocation( ctx );
        mainFirstTime( loc );
        setLastLoc( loc );
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
            throw exception_with_context(
                __FILE__, __LINE__, __func__,
                std::format( "{}Internal error: Unsupported floating point declaration size.\n",
                             formatLocation( loc ) ) );
        }
    }

    void MLIRListener::enterStringDeclare( ToyParser::StringDeclareContext *ctx )
    {
        assert( ctx );
        mlir::Location loc = getLocation( ctx );
        mainFirstTime( loc );
        setLastLoc( loc );
        assert( ctx->IDENTIFIER() );
        std::string varName = ctx->IDENTIFIER()->getText();
        ToyParser::ArrayBoundsExpressionContext *arrayBounds = ctx->arrayBoundsExpression();
        assert( arrayBounds );

        registerDeclaration( loc, varName, tyI8, arrayBounds );
    }

    mlir::Value MLIRListener::parsePredicate( mlir::Location loc, ToyParser::BooleanValueContext *booleanValue )
    {
        // booleanValue
        //   : booleanElement | (binaryElement predicateOperator binaryElement)
        //   ;

        mlir::Value conditionPredicate{};

        assert( booleanValue );
        if ( ToyParser::BooleanElementContext * boolElement = booleanValue->booleanElement() )
        {
            ToyParser::BooleanLiteralContext * lit = boolElement->booleanLiteral();
            conditionPredicate = buildNonStringUnaryExpression( loc, lit ? lit->BOOLEAN_PATTERN() : nullptr,
                                                                lit ? lit->INTEGER_PATTERN() : nullptr, nullptr,
                                                                boolElement->scalarOrArrayElement(), nullptr );
        }
        else
        {
            // binaryElement : numericLiteral | unaryOperator? IDENTIFIER

            std::vector<ToyParser::BinaryElementContext *> operands = booleanValue->binaryElement();
            if ( operands.size() == 2 )
            {
                mlir::Value lhsValue;
                mlir::Value rhsValue;
                ToyParser::BinaryElementContext * lhs = operands[0];
                assert( lhs );
                ToyParser::BinaryElementContext * rhs = operands[1];
                assert( rhs );

                ToyParser::NumericLiteralContext * llit = lhs->numericLiteral();
                lhsValue = buildNonStringUnaryExpression( loc, nullptr,    // booleanNode
                                                          llit ? llit->INTEGER_PATTERN() : nullptr,
                                                          llit ? llit->FLOAT_PATTERN() : nullptr,
                                                          lhs->scalarOrArrayElement(), nullptr );

                ToyParser::NumericLiteralContext * rlit = rhs->numericLiteral();
                rhsValue = buildNonStringUnaryExpression( loc, nullptr,    // booleanNode
                                                          rlit ? rlit->INTEGER_PATTERN() : nullptr,
                                                          rlit ? rlit->FLOAT_PATTERN() : nullptr,
                                                          lhs->scalarOrArrayElement(), nullptr );

                assert( booleanValue );
                ToyParser::PredicateOperatorContext * op = booleanValue->predicateOperator();
                assert( op );
                if ( op->LESSTHAN_TOKEN() )
                {
                    conditionPredicate = builder.create<toy::LessOp>( loc, tyI1, lhsValue, rhsValue ).getResult();
                }
                else if ( op->GREATERTHAN_TOKEN() )
                {
                    conditionPredicate = builder.create<toy::LessOp>( loc, tyI1, rhsValue, lhsValue ).getResult();
                }
                else if ( op->LESSEQUAL_TOKEN() )
                {
                    conditionPredicate = builder.create<toy::LessEqualOp>( loc, tyI1, lhsValue, rhsValue ).getResult();
                }
                else if ( op->GREATEREQUAL_TOKEN() )
                {
                    conditionPredicate = builder.create<toy::LessEqualOp>( loc, tyI1, rhsValue, lhsValue ).getResult();
                }
                else if ( op->EQUALITY_TOKEN() )
                {
                    conditionPredicate = builder.create<toy::EqualOp>( loc, tyI1, lhsValue, rhsValue ).getResult();
                }
                else if ( op->NOTEQUAL_TOKEN() )
                {
                    conditionPredicate = builder.create<toy::NotEqualOp>( loc, tyI1, lhsValue, rhsValue ).getResult();
                }
                else
                {
                    throw exception_with_context(
                        __FILE__, __LINE__, __func__,
                        std::format( "{}error: Unsupported binary operator in if condition.\n",
                                     formatLocation( loc ) ) );
                }
            }
            else
            {
                throw exception_with_context(
                    __FILE__, __LINE__, __func__,
                    std::format( "{}error: Only binary operators supported in if condition (for now).\n",
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
            throw exception_with_context(
                __FILE__, __LINE__, __func__,
                std::format( "{}internal error: if condition must be i1\n", formatLocation( loc ) ) );
        }

        return conditionPredicate;
    }

    void MLIRListener::enterIfelifelse( ToyParser::IfelifelseContext *ctx )
    {
        assert( ctx );
        mlir::Location loc = getLocation( ctx );
        mainFirstTime( loc );

        ToyParser::IfStatementContext * theIf = ctx->ifStatement();
        assert( theIf );
        ToyParser::BooleanValueContext *booleanValue = theIf->booleanValue();
        assert( booleanValue );

        LLVM_DEBUG( {
            std::vector<ToyParser::StatementContext *> statements = theIf->statement();
            std::cout << std::format( "IF: ({})", booleanValue->getText() );
            for ( ToyParser::StatementContext * s : statements )
            {
                std::cout << std::format( " STATEMENT: {}", s->getText() );
            }
            std::cout << "\n";
        } );

        mlir::Value conditionPredicate = MLIRListener::parsePredicate( loc, booleanValue );

        insertionPointStack.push_back( builder.saveInsertionPoint() );

        mlir::scf::IfOp ifOp = builder.create<mlir::scf::IfOp>( loc, conditionPredicate );

        mlir::Block &thenBlock = ifOp.getThenRegion().front();
        builder.setInsertionPointToStart( &thenBlock );
    }

    void MLIRListener::exitIfStatement( ToyParser::IfStatementContext *ctx )
    {
        assert( ctx );
        mlir::Location loc = getLocation( ctx );
        // All statements in the if-body have now been processed by their own enter/exit callbacks, accumulated
        // into an scf.if then region.

        antlr4::tree::ParseTree *parent = ctx->parent;
        assert( parent );
        ToyParser::IfelifelseContext *ifElifElse = dynamic_cast<ToyParser::IfelifelseContext *>( parent );

        if ( !ifElifElse )
        {
            throw exception_with_context( __FILE__, __LINE__, __func__,
                                          std::format( "{}IfStatement parent is not IfelifelseContext: ctx: {}\n",
                                                       formatLocation( loc ), ctx->getText() ) );
        }

        std::vector<ToyParser::ElifStatementContext *> elifs = ifElifElse->elifStatement();
        if ( elifs.size() )
        {
            throw exception_with_context( __FILE__, __LINE__, __func__,
                                          std::format( "{}ELIF NYI: {}\n", formatLocation( loc ), ctx->getText() ) );
        }

        ToyParser::ElseStatementContext * elseCtx = ifElifElse->elseStatement();
        if ( elseCtx )
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
                throw exception_with_context(
                    __FILE__, __LINE__, __func__,
                    std::format( "{}internal error: Could not find scf.if op corresponding to this if statement\n",
                                 formatLocation( loc ), ctx->getText() ) );
            }

            mlir::Region &elseRegion = ifOp.getElseRegion();
            if ( !elseRegion.empty() )
            {
                throw exception_with_context( __FILE__, __LINE__, __func__,
                                              std::format( "{}internal error: Expected empty else region\n",
                                                           formatLocation( loc ), ctx->getText() ) );
            }

            elseRegion.emplaceBlock();    // creates one empty block

            mlir::Block &elseBlock = elseRegion.front();
            builder.setInsertionPointToStart( &elseBlock );
        }
        else
        {
            // Restore EXACTLY where we were before creating the scf.if
            // This places new ops right AFTER the scf.if
            builder.restoreInsertionPoint( insertionPointStack.back() );
            insertionPointStack.pop_back();
        }

        // LLVM_DEBUG( { llvm::errs() << "exitIfStatement module dump:\n"; mod->dump(); } );
    }

    void MLIRListener::exitElseStatement( ToyParser::ElseStatementContext *ctx )
    {
        assert( ctx );
        mlir::Location loc = getLocation( ctx, true );
        builder.create<mlir::scf::YieldOp>( loc );

        builder.restoreInsertionPoint( insertionPointStack.back() );
        insertionPointStack.pop_back();

        // LLVM_DEBUG( { llvm::errs() << "exitElseStatement module dump:\n"; mod->dump(); } );
    }

    void MLIRListener::enterFor( ToyParser::ForContext *ctx )
    {
        assert( ctx );
        mlir::Location loc = getLocation( ctx, true );
        mainFirstTime( loc );
        setLastLoc( loc );

        LLVM_DEBUG( { llvm::errs() << std::format( "For: {}\n", ctx->getText() ); } );

        assert( ctx->IDENTIFIER() );
        std::string varName = ctx->IDENTIFIER()->getText();
        variable_state varState = getVarState( varName );
        if ( varState == variable_state::undeclared )
        {
            throw exception_with_context(
                __FILE__, __LINE__, __func__,
                std::format( "{}error: Variable {} not declared in PRINT\n", formatLocation( loc ), varName ) );
        }

        mlir::SymbolRefAttr symRef = mlir::SymbolRefAttr::get( &dialect.context, varName );
        mlir::NamedAttribute varNameAttr( builder.getStringAttr( "var_name" ), symRef );

        assert( ctx->forStart() );
        assert( ctx->forEnd() );
        ToyParser::ParameterContext *pStart = ctx->forStart()->parameter();
        ToyParser::ParameterContext *pEnd = ctx->forEnd()->parameter();
        ToyParser::ParameterContext *pStep{};
        if ( ToyParser::ForStepContext * st = ctx->forStep() )
        {
            pStep = st->parameter();
        }

        mlir::Value start;
        mlir::Value end;
        mlir::Value step;

        toy::DeclareOp declareOp = lookupDeclareForVar( loc, varName );
        mlir::Type elemType = declareOp.getTypeAttr().getValue();

        if ( pStart )
        {
            ToyParser::LiteralContext * lit = pStart->literal();
            start = buildNonStringUnaryExpression( loc, lit ? lit->BOOLEAN_PATTERN() : nullptr,
                                                   lit ? lit->INTEGER_PATTERN() : nullptr,
                                                   lit ? lit->FLOAT_PATTERN() : nullptr, pStart->scalarOrArrayElement(),
                                                   lit ? lit->STRING_PATTERN() : nullptr );

            start = castOpIfRequired( loc, start, elemType );
        }
        else
        {
            throw exception_with_context(
                __FILE__, __LINE__, __func__,
                std::format( "{}FOR loop: expected start index: {}\n", formatLocation( loc ), ctx->getText() ) );
        }

        if ( pEnd )
        {
            ToyParser::LiteralContext * lit = pEnd->literal();
            end = buildNonStringUnaryExpression( loc, lit ? lit->BOOLEAN_PATTERN() : nullptr,
                                                 lit ? lit->INTEGER_PATTERN() : nullptr,
                                                 lit ? lit->FLOAT_PATTERN() : nullptr, pEnd->scalarOrArrayElement(),
                                                 lit ? lit->STRING_PATTERN() : nullptr );

            end = castOpIfRequired( loc, end, elemType );
        }
        else
        {
            throw exception_with_context(
                __FILE__, __LINE__, __func__,
                std::format( "{}FOR loop: expected end index: {}\n", formatLocation( loc ), ctx->getText() ) );
        }

        if ( pStep )
        {
            ToyParser::LiteralContext * lit = pStep->literal();
            step = buildNonStringUnaryExpression( loc, lit ? lit->BOOLEAN_PATTERN() : nullptr,
                                                  lit ? lit->INTEGER_PATTERN() : nullptr,
                                                  lit ? lit->FLOAT_PATTERN() : nullptr, pStep->scalarOrArrayElement(),
                                                  lit ? lit->STRING_PATTERN() : nullptr );
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
        builder.create<toy::AssignOp>( loc, mlir::TypeRange{}, mlir::ValueRange{ inductionVar },
                                       llvm::ArrayRef<mlir::NamedAttribute>{ varNameAttr } );
        setVarState( currentFuncName, varName, variable_state::assigned );
    }

    void MLIRListener::exitFor( ToyParser::ForContext *ctx )
    {
        assert( ctx );
        builder.restoreInsertionPoint( insertionPointStack.back() );
        insertionPointStack.pop_back();
    }

    void MLIRListener::enterPrint( ToyParser::PrintContext *ctx )
    {
        assert( ctx );
        mlir::Location loc = getLocation( ctx );
        mainFirstTime( loc );
        setLastLoc( loc );

        mlir::Type varType;

        ToyParser::ScalarOrArrayElementContext *scalarOrArrayElement = ctx->scalarOrArrayElement();
        if ( scalarOrArrayElement )
        {
            tNode * varNameObject = scalarOrArrayElement->IDENTIFIER();
            assert( varNameObject );
            std::string varName = varNameObject->getText();
            variable_state varState = getVarState( varName );
            if ( varState == variable_state::undeclared )
            {
                throw exception_with_context(
                    __FILE__, __LINE__, __func__,
                    std::format( "{}error: Variable {} not declared in PRINT\n", formatLocation( loc ), varName ) );
            }
            if ( varState != variable_state::assigned )
            {
                throw exception_with_context(
                    __FILE__, __LINE__, __func__,
                    std::format( "{}error: Variable {} not assigned in PRINT\n", formatLocation( loc ), varName ) );
            }

            toy::DeclareOp declareOp = lookupDeclareForVar( loc, varName );

            mlir::Type elemType = declareOp.getTypeAttr().getValue();
            mlir::Value optIndexValue{};
            if ( ToyParser::IndexExpressionContext *indexExpr = scalarOrArrayElement->indexExpression() )
            {
                mlir::Value indexValue{};
                indexValue = buildNonStringUnaryExpression( loc, nullptr, indexExpr->INTEGER_PATTERN(), nullptr,
                                                            scalarOrArrayElement, nullptr );

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
            toy::LoadOp value = builder.create<toy::LoadOp>( loc, varType, symRef, optIndexValue );
            builder.create<toy::PrintOp>( loc, value );
        }
        else if ( tNode * theString = ctx->STRING_PATTERN() )
        {
            assert( theString );
            std::string s = stripQuotes( loc, theString->getText() );
            mlir::StringAttr strAttr = builder.getStringAttr( s );

            toy::StringLiteralOp stringLiteral = builder.create<toy::StringLiteralOp>( loc, tyPtr, strAttr );

            builder.create<toy::PrintOp>( loc, stringLiteral );
        }
        else
        {
            throw exception_with_context(
                __FILE__, __LINE__, __func__,
                std::format( "{}error: unexpected print context {}\n", formatLocation( loc ), ctx->getText() ) );
        }
    }

    void MLIRListener::enterGet( ToyParser::GetContext *ctx )
    {
        assert( ctx );
        mlir::Location loc = getLocation( ctx );
        mainFirstTime( loc );
        setLastLoc( loc );

        ToyParser::ScalarOrArrayElementContext *scalarOrArrayElement = ctx->scalarOrArrayElement();
        if ( scalarOrArrayElement )
        {
            tNode * varNameObject = scalarOrArrayElement->IDENTIFIER();
            assert( varNameObject );
            std::string varName = varNameObject->getText();
            variable_state varState = getVarState( varName );
            if ( varState == variable_state::undeclared )
            {
                throw exception_with_context(
                    __FILE__, __LINE__, __func__,
                    std::format( "{}error: Variable {} not declared in GET\n", formatLocation( loc ), varName ) );
            }

            toy::DeclareOp declareOp = lookupDeclareForVar( loc, varName );

            mlir::Type elemType = declareOp.getTypeAttr().getValue();
            mlir::Value optIndexValue{};
            if ( ToyParser::IndexExpressionContext *indexExpr = scalarOrArrayElement->indexExpression() )
            {
                mlir::Value indexValue{};
                indexValue = buildNonStringUnaryExpression( loc, nullptr, indexExpr->INTEGER_PATTERN(), nullptr,
                                                            scalarOrArrayElement, nullptr );

                optIndexValue = indexTypeCast( loc, indexValue );
            }
            else if ( declareOp.getSizeAttr() )
            {
                throw exception_with_context( __FILE__, __LINE__, __func__,
                                              std::format( "{}error: attempted GET to string literal?{}\n",
                                                           formatLocation( loc ), ctx->getText() ) );
            }
            else
            {
                // Scalar: load the value
            }

            mlir::SymbolRefAttr symRef = mlir::SymbolRefAttr::get( &dialect.context, varName );

            toy::GetOp resultValue = builder.create<toy::GetOp>( loc, elemType );
            builder.create<toy::AssignOp>( loc, symRef, optIndexValue, resultValue );

            setVarState( currentFuncName, varName, variable_state::assigned );
        }
        else
        {
            throw exception_with_context(
                __FILE__, __LINE__, __func__,
                std::format( "{}error: unexpected get context {}\n", formatLocation( loc ), ctx->getText() ) );
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

    template <class Literal>
    void MLIRListener::processReturnLike( mlir::Location loc, Literal *lit,
                                          ToyParser::ScalarOrArrayElementContext *scalarOrArrayElement,
                                          tNode *boolNode )
    {
        // Handle the dummy ReturnOp originally inserted in the FuncOp's block
        mlir::Block *currentBlock = builder.getInsertionBlock();
        assert( currentBlock );
        assert( !currentBlock->empty() );
        mlir::Operation *parentOp = currentBlock->getParentOp();
        assert( parentOp );
        if ( !isa<toy::ScopeOp>( parentOp ) )
        {
            throw exception_with_context(
                __FILE__, __LINE__, __func__,
                std::format( "{}error: RETURN statement must be inside a toy.scope\n", formatLocation( loc ) ) );
        }

        mlir::func::FuncOp func = parentOp->getParentOfType<mlir::func::FuncOp>();
        llvm::ArrayRef<mlir::Type> returnType = func.getFunctionType().getResults();

        if ( !isa<toy::ReturnOp>( currentBlock->getTerminator() ) )
        {
            throw exception_with_context(
                __FILE__, __LINE__, __func__,
                std::format( "{}internal error: Expected toy::ReturnOp terminator\n", formatLocation( loc ) ) );
        }

        // Erase existing toy::ReturnOp and its constant
        mlir::Operation *existingExit = currentBlock->getTerminator();
        assert( existingExit );
        mlir::Operation *constantOp{};
        if ( existingExit->getNumOperands() > 0 )
        {
            constantOp = existingExit->getOperand( 0 ).getDefiningOp();
        }
        existingExit->erase();

        if ( constantOp && mlir::isa<mlir::arith::ConstantOp>( constantOp ) )
        {
            constantOp->erase();
        }

        // Set insertion point to func::FuncOp block
        builder.setInsertionPointToEnd( currentBlock );

        mlir::Value value{};

        // always regenerate the RETURN/EXIT so that we have the terminator location set properly (not the function body
        // start location that was used to create the dummy toy.ReturnOp that we rewrite here.)
        if ( lit || scalarOrArrayElement || boolNode )
        {
            value =
                buildNonStringUnaryExpression( loc, boolNode, lit ? lit->INTEGER_PATTERN() : nullptr,
                                               lit ? lit->FLOAT_PATTERN() : nullptr, scalarOrArrayElement, nullptr );

            // Apply type conversions to match func::FuncOp return type.  This is adapted from AssignOpLowering, but
            // uses arith dialect operations instead of LLVM dialect
            if ( !returnType.empty() )
            {
                value = castOpIfRequired( loc, value, returnType[0] );
            }
        }
        else
        {
            if ( !returnType.empty() )
            {
                if ( mlir::IntegerType intType = mlir::dyn_cast<mlir::IntegerType>( returnType[0] ) )
                {
                    unsigned width = intType.getWidth();
                    value = builder.create<mlir::arith::ConstantIntOp>( loc, 0, width );
                }
                else if ( mlir::FloatType floatType = mlir::dyn_cast<mlir::FloatType>( returnType[0] ) )
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
                        throw exception_with_context(
                            __FILE__, __LINE__, __func__,
                            std::format( "{}Support for FloatType w/ size other than 32/64 is not implemented: {}\n",
                                         formatLocation( loc ), w ) );
                    }
                }
                else
                {
                    throw exception_with_context(
                        __FILE__, __LINE__, __func__,
                        std::format( "{}Return support for non-(IntegerType,FloatType) is not implemented\n",
                                     formatLocation( loc ) ) );
                }
            }
        }

        // Create new ReturnOp with user specified value:
        if ( value )
        {
            builder.create<toy::ReturnOp>( loc, mlir::ValueRange{ value } );
        }
        else
        {
            builder.create<toy::ReturnOp>( loc, mlir::ValueRange{} );
        }

        markExplicitTerminator();
    }

    void MLIRListener::enterReturnStatement( ToyParser::ReturnStatementContext *ctx )
    {
        assert( ctx );
        mlir::Location loc = getLocation( ctx );
        mainFirstTime( loc );
        setLastLoc( loc );

        ToyParser::LiteralContext * lit = ctx->literal();
        processReturnLike<ToyParser::LiteralContext>( loc, lit, ctx->scalarOrArrayElement(),
                                                      lit ? lit->BOOLEAN_PATTERN() : nullptr );
    }

    void MLIRListener::enterExitStatement( ToyParser::ExitStatementContext *ctx )
    {
        assert( ctx );
        mlir::Location loc = getLocation( ctx );
        mainFirstTime( loc );
        setLastLoc( loc );

        ToyParser::NumericLiteralContext * lit = ctx->numericLiteral();

        processReturnLike<ToyParser::NumericLiteralContext>( loc, lit, ctx->scalarOrArrayElement(), nullptr );
    }

    void MLIRListener::enterAssignment( ToyParser::AssignmentContext *ctx )
    {
        assert( ctx );
        assignmentTargetValid = true;
        mlir::Location loc = getLocation( ctx );
        mainFirstTime( loc );
        setLastLoc( loc );
        ToyParser::ScalarOrArrayElementContext * lhs = ctx->scalarOrArrayElement();
        assert( lhs );
        assert( lhs->IDENTIFIER() );
        currentVarName = lhs->IDENTIFIER()->getText();

        ToyParser::IndexExpressionContext *indexExpr = lhs->indexExpression();

        currentIndexExpr = mlir::Value();

        if ( indexExpr )
        {
            currentIndexExpr =
                buildNonStringUnaryExpression( loc, nullptr, indexExpr->INTEGER_PATTERN(), nullptr, lhs, nullptr );
        }

        variable_state varState = getVarState( currentVarName );
        if ( varState == variable_state::declared )
        {
            setVarState( currentFuncName, currentVarName, variable_state::assigned );
        }
        else if ( varState == variable_state::undeclared )
        {
            throw exception_with_context( __FILE__, __LINE__, __func__,
                                          std::format( "{}error: Variable {} not declared in assignment\n",
                                                       formatLocation( loc ), currentVarName ) );

            assignmentTargetValid = false;
        }
        currentAssignLoc = loc;
        callIsHandled = false;
    }

    void MLIRListener::exitAssignment( ToyParser::AssignmentContext *ctx )
    {
        assert( ctx );
        callIsHandled = false;
    }

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
            throw exception_with_context(
                __FILE__, __LINE__, __func__,
                std::format( "{}NYI: indexTypeCast for types other than i64\n", formatLocation( loc ) ) );
        }

        return builder.create<mlir::arith::IndexCastOp>( loc, indexTy, val );
    }

    void MLIRListener::enterRhs( ToyParser::RhsContext *ctx )
    {
        assert( ctx );
        if ( !assignmentTargetValid )
        {
            return;
        }
        mlir::Location loc = getLocation( ctx );
        mainFirstTime( loc );
        setLastLoc( loc );
        mlir::Value resultValue;

        toy::DeclareOp declareOp = lookupDeclareForVar( loc, currentVarName );
        mlir::TypeAttr typeAttr = declareOp.getTypeAttr();
        mlir::Type opType = typeAttr.getValue();

        mlir::Value lhsValue;
        size_t bsz = ctx->binaryElement().size();
        std::string s;

        if ( bsz == 0 )
        {
            mlir::Value lhsValue;

            ToyParser::LiteralContext * lit = ctx->literal();

            if ( ToyParser::CallContext * call = ctx->call() )
            {
                callIsHandled = true;
                lhsValue = handleCall( call );
            }
            else
            {
                lhsValue =
                    buildUnaryExpression( loc, lit ? lit->BOOLEAN_PATTERN() : nullptr,
                                          lit ? lit->INTEGER_PATTERN() : nullptr, lit ? lit->FLOAT_PATTERN() : nullptr,
                                          ctx->scalarOrArrayElement(), lit ? lit->STRING_PATTERN() : nullptr, s );
            }

            resultValue = lhsValue;
            if ( ToyParser::UnaryOperatorContext * unaryOp = ctx->unaryOperator() )
            {
                std::string opText = unaryOp->getText();
                if ( opText == "-" )
                {
                    toy::NegOp op = builder.create<toy::NegOp>( loc, opType, lhsValue );
                    resultValue = op.getResult();

                    if ( s.length() )
                    {
                        throw exception_with_context(
                            __FILE__, __LINE__, __func__,
                            std::format( "{}internal error: Unexpected string literal for negation operation\n",
                                         formatLocation( loc ) ) );
                    }
                }
                else if ( opText == "NOT" )
                {
                    mlir::arith::ConstantIntOp rhsValue = builder.create<mlir::arith::ConstantIntOp>( loc, 0, 64 );

                    toy::EqualOp b = builder.create<toy::EqualOp>( loc, opType, lhsValue, rhsValue );
                    resultValue = b.getResult();

                    if ( s.length() )
                    {
                        throw exception_with_context(
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
                throw exception_with_context(
                    __FILE__, __LINE__, __func__,
                    std::format( "{}internal error: binaryElement size != 2\n", formatLocation( loc ) ) );
            }

            ToyParser::BinaryElementContext * lhs = ctx->binaryElement()[0];
            assert( lhs );
            ToyParser::BinaryElementContext * rhs = ctx->binaryElement()[1];
            assert( rhs );
            std::string opText = ctx->binaryOperator()->getText();

            ToyParser::NumericLiteralContext * llit = lhs->numericLiteral();
            lhsValue = buildNonStringUnaryExpression( loc, nullptr,    // booleanNode
                                                      llit ? llit->INTEGER_PATTERN() : nullptr,
                                                      llit ? llit->FLOAT_PATTERN() : nullptr,
                                                      lhs->scalarOrArrayElement(), nullptr );

            mlir::Value rhsValue;
            ToyParser::NumericLiteralContext * rlit = rhs->numericLiteral();
            rhsValue = buildNonStringUnaryExpression( loc, nullptr,    // booleanNode
                                                      rlit ? rlit->INTEGER_PATTERN() : nullptr,
                                                      rlit ? rlit->FLOAT_PATTERN() : nullptr,
                                                      rhs->scalarOrArrayElement(), nullptr );

            // Create the binary operator (supports +, -, *, /)
            if ( opText == "+" )
            {
                toy::AddOp b = builder.create<toy::AddOp>( loc, opType, lhsValue, rhsValue );
                resultValue = b.getResult();
            }
            else if ( opText == "-" )
            {
                toy::SubOp b = builder.create<toy::SubOp>( loc, opType, lhsValue, rhsValue );
                resultValue = b.getResult();
            }
            else if ( opText == "*" )
            {
                toy::MulOp b = builder.create<toy::MulOp>( loc, opType, lhsValue, rhsValue );
                resultValue = b.getResult();
            }
            else if ( opText == "/" )
            {
                toy::DivOp b = builder.create<toy::DivOp>( loc, opType, lhsValue, rhsValue );
                resultValue = b.getResult();
            }
            else if ( opText == "<" )
            {
                toy::LessOp b = builder.create<toy::LessOp>( loc, opType, lhsValue, rhsValue );
                resultValue = b.getResult();
            }
            else if ( opText == ">" )
            {
                toy::LessOp b = builder.create<toy::LessOp>( loc, opType, rhsValue, lhsValue );
                resultValue = b.getResult();
            }
            else if ( opText == "<=" )
            {
                toy::LessEqualOp b = builder.create<toy::LessEqualOp>( loc, opType, lhsValue, rhsValue );
                resultValue = b.getResult();
            }
            else if ( opText == ">=" )
            {
                toy::LessEqualOp b = builder.create<toy::LessEqualOp>( loc, opType, rhsValue, lhsValue );
                resultValue = b.getResult();
            }
            else if ( opText == "EQ" )
            {
                toy::EqualOp b = builder.create<toy::EqualOp>( loc, opType, lhsValue, rhsValue );
                resultValue = b.getResult();
            }
            else if ( opText == "NE" )
            {
                toy::NotEqualOp b = builder.create<toy::NotEqualOp>( loc, opType, lhsValue, rhsValue );
                resultValue = b.getResult();
            }
            else if ( opText == "AND" )
            {
                toy::AndOp b = builder.create<toy::AndOp>( loc, opType, lhsValue, rhsValue );
                resultValue = b.getResult();
            }
            else if ( opText == "OR" )
            {
                toy::OrOp b = builder.create<toy::OrOp>( loc, opType, lhsValue, rhsValue );
                resultValue = b.getResult();
            }
            else if ( opText == "XOR" )
            {
                toy::XorOp b = builder.create<toy::XorOp>( loc, opType, lhsValue, rhsValue );
                resultValue = b.getResult();
            }
            else
            {
                throw exception_with_context(
                    __FILE__, __LINE__, __func__,
                    std::format( "{}error: Invalid binary operator {}\n", formatLocation( loc ), opText ) );
            }
        }

        if ( currentVarName.empty() )
        {
            throw exception_with_context( __FILE__, __LINE__, __func__,
                                          std::format( "{}error: currentVarName not set!\n", formatLocation( loc ) ) );
        }

        mlir::SymbolRefAttr symRef = mlir::SymbolRefAttr::get( &dialect.context, currentVarName );
        if ( s.length() )
        {
            mlir::StringAttr strAttr = builder.getStringAttr( s );

            toy::StringLiteralOp stringLiteral = builder.create<toy::StringLiteralOp>( loc, tyPtr, strAttr );

            mlir::NamedAttribute varNameAttr( builder.getStringAttr( "var_name" ), symRef );

            builder.create<toy::AssignOp>( loc, mlir::TypeRange{}, mlir::ValueRange{ stringLiteral },
                                           llvm::ArrayRef<mlir::NamedAttribute>{ varNameAttr } );
        }
        else
        {
            if ( currentIndexExpr )
            {
                mlir::Value i = indexTypeCast( loc, currentIndexExpr );

                toy::AssignOp assign = builder.create<toy::AssignOp>( loc, symRef, i, resultValue );

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

                builder.create<toy::AssignOp>( loc, mlir::TypeRange{}, mlir::ValueRange{ resultValue },
                                               llvm::ArrayRef<mlir::NamedAttribute>{ varNameAttr } );
            }
        }

        setVarState( currentFuncName, currentVarName, variable_state::assigned );
        currentVarName.clear();
        currentIndexExpr = mlir::Value();
    }
}    // namespace toy

// vim: et ts=4 sw=4
