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
    DialectCtx::DialectCtx()
    {
        context.getOrLoadDialect<toy::ToyDialect>();
        context.getOrLoadDialect<mlir::func::FuncDialect>();
        context.getOrLoadDialect<mlir::arith::ArithDialect>();
        context.getOrLoadDialect<mlir::memref::MemRefDialect>();
        context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
    }

    toy::DeclareOp MLIRListener::lookupDeclareForVar( const std::string &varName )
    {
        mlir::func::FuncOp funcOp = getFuncOp( currentFuncName );

        // Get the single block in the func::FuncOp's region
        auto &funcBlock = funcOp.getBody().front();

        toy::ScopeOp scopeOp{};
        for ( auto &op : funcBlock )
        {
            if ( mlir::isa<toy::ScopeOp>( &op ) )
            {
                scopeOp = mlir::dyn_cast<toy::ScopeOp>( &op );
#if 0
                LLVM_DEBUG({
                    llvm::errs() << std::format("Found toy::ScopeOp while looking for symbol {}\n", varName);
                    scopeOp.dump();
                });
#endif
                break;
            }
        }

        assert( scopeOp );
#if 1
        LLVM_DEBUG( {
            llvm::errs() << std::format( "Lookup symbol {} in parent function:\n", varName );
            scopeOp->dump();
        } );
#endif

        auto *symbolOp = mlir::SymbolTable::lookupSymbolIn( scopeOp, varName );
        if ( !symbolOp )
        {
            throw exception_with_context( __FILE__, __LINE__, __func__,
                                          std::format( "Undeclared variable {}", varName ) );
        }

        auto declareOp = mlir::dyn_cast<toy::DeclareOp>( symbolOp );
        if ( !declareOp )
        {
            throw exception_with_context( __FILE__, __LINE__, __func__,
                                          std::format( "Undeclared variable {}", varName ) );
        }

        return declareOp;
    }

    inline mlir::Location MLIRListener::getLocation( antlr4::ParserRuleContext *ctx )
    {
        size_t line = 1;
        size_t col = 0;
        if ( ctx )
        {
            line = ctx->getStart()->getLine();
            col = ctx->getStart()->getCharPositionInLine();
        }

        auto loc = mlir::FileLineColLoc::get( builder.getStringAttr( filename ), line, col + 1 );
        lastLocation = loc;

        return loc;
    }

    inline std::string MLIRListener::formatLocation( mlir::Location loc )
    {
        if ( auto fileLoc = mlir::dyn_cast<mlir::FileLineColLoc>( loc ) )
        {
            return std::format( "{}:{}:{}: ", fileLoc.getFilename().str(), fileLoc.getLine(), fileLoc.getColumn() );
        }
        return "";
    }

    inline theTypes getCompilerType( mlir::Type mtype )
    {
        theTypes ty = theTypes::unknown;

        if ( auto intType = mlir::dyn_cast<mlir::IntegerType>( mtype ) )
        {
            switch ( intType.getWidth() )
            {
                case 1:
                    ty = theTypes::boolean;
                    break;
                case 8:
                    ty = theTypes::integer8;
                    break;
                case 16:
                    ty = theTypes::integer16;
                    break;
                case 32:
                    ty = theTypes::integer32;
                    break;
                case 64:
                    ty = theTypes::integer64;
                    break;
                default:
                    throw exception_with_context( __FILE__, __LINE__, __func__,
                                                  "internal error: unexpected integer width" );
            }
        }
        else if ( auto floatType = mlir::dyn_cast<mlir::FloatType>( mtype ) )
        {
            switch ( floatType.getWidth() )
            {
                case 32:
                    ty = theTypes::float32;
                    break;
                case 64:
                    ty = theTypes::float64;
                    break;
                default:
                    throw exception_with_context( __FILE__, __LINE__, __func__,
                                                  "internal error: unexpected float width" );
            }
        }
        else
        // if ( auto stringType = mlir::dyn_cast<mlir::StringAttr>( mtype ) ) // hack
        {
            ty = theTypes::string;
        }

        if ( ty == theTypes::unknown )
        {
            throw exception_with_context( __FILE__, __LINE__, __func__, "internal error: unhandled type" );
        }

        return ty;
    }

    inline std::string stripQuotes( const std::string &input )
    {
        assert( input.size() >= 2 );
        assert( input.front() == '"' );
        assert( input.back() == '"' );

        return input.substr( 1, input.size() - 2 );
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

    // \retval true if error
    inline void MLIRListener::buildUnaryExpression( tNode *booleanNode, tNode *integerNode, tNode *floatNode,
                                                    tNode *variableNode, tNode *stringNode, mlir::Location loc,
                                                    mlir::Value &value, std::string &s )
    {
        if ( booleanNode )
        {
            int val;
            auto bv = booleanNode->getText();
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
            value = builder.create<mlir::arith::ConstantFloatOp>( loc, apVal, tyF64 );
        }
        else if ( variableNode )
        {
            auto varName = variableNode->getText();

            auto varState = getVarState( varName );

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

            auto declareOp = lookupDeclareForVar( varName );

            mlir::Type varType = declareOp.getTypeAttr().getValue();
            auto symRef = mlir::SymbolRefAttr::get( &dialect.context, varName );
            value = builder.create<toy::LoadOp>( loc, varType, symRef );
        }
        else if ( stringNode )
        {
            s = stripQuotes( stringNode->getText() );
        }
        else
        {
            throw exception_with_context( __FILE__, __LINE__, __func__,
                                          std::format( "{}error: Invalid operand\n", formatLocation( loc ) ) );
        }
    }

    // \retval true if error
    inline void MLIRListener::registerDeclaration( mlir::Location loc, const std::string &varName, mlir::Type ty,
                                                   ToyParser::ArrayBoundsExpressionContext *arrayBounds )
    {
        auto varState = getVarState( varName );
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
            auto index = arrayBounds->INTEGER_PATTERN();
            arraySize = std::stoi( index->getText() );
        }

        auto strAttr = builder.getStringAttr( varName );
        if ( arraySize )
        {
            auto sizeAttr = builder.getI64IntegerAttr( arraySize );
            auto dcl = builder.create<toy::DeclareOp>( loc, mlir::TypeAttr::get( ty ), sizeAttr, /*parameter=*/nullptr,
                                                       nullptr );
            dcl->setAttr( "sym_name", strAttr );
        }
        else
        {
            auto dcl = builder.create<toy::DeclareOp>( loc, mlir::TypeAttr::get( ty ), nullptr, /*parameter=*/nullptr,
                                                       nullptr );
            dcl->setAttr( "sym_name", strAttr );
        }

        // For test purposes to verify that symbol lookup for varName worked right after the DeclareOp build call:
        // auto ddcl = lookupDeclareForVar( varName );
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

        auto ctx = builder.getContext();
        tyVoid = mlir::LLVM::LLVMVoidType::get( ctx );
        tyPtr = mlir::LLVM::LLVMPointerType::get( ctx );
    }

    void MLIRListener::createScope( mlir::Location loc, mlir::func::FuncOp funcOp, const std::string &funcName,
                                    const std::vector<std::string> &paramNames )
    {
        auto &block = *funcOp.addEntryBlock();
        builder.setInsertionPointToStart( &block );

        // Create Toy::ScopeOp with empty operands and results
        auto scopeOp = builder.create<toy::ScopeOp>( loc, mlir::TypeRange{}, mlir::ValueRange{} );
        builder.create<toy::YieldOp>( loc );

        auto &scopeBlock = scopeOp.getBody().emplaceBlock();

        builder.setInsertionPointToStart( &scopeBlock );

        for ( size_t i = 0; i < funcOp.getNumArguments() && i < paramNames.size(); ++i )
        {
            auto argType = funcOp.getArgument( i ).getType();
            LLVM_DEBUG( {
                llvm::errs() << std::format( "function {}: parameter{}:\n", funcName, i );
                funcOp.getArgument( i ).dump();
            } );
            auto strAttr = builder.getStringAttr( paramNames[i] );
            auto dcl = builder.create<toy::DeclareOp>( loc, mlir::TypeAttr::get( argType ), /*size=*/nullptr,
                                                       builder.getUnitAttr(), builder.getI64IntegerAttr( i ) );
            dcl->setAttr( "sym_name", strAttr );
        }

        // Insert a default toy::ReturnOp terminator (with no operands, or default zero return for scalar return
        // functions, like main). This will be replaced later with an toy::ReturnOp with the actual return code if
        // desired.
        auto returnType = funcOp.getFunctionType().getResults();
        mlir::Operation *returnOp = nullptr;
        if ( !returnType.empty() )
        {
            auto zero = builder.create<mlir::arith::ConstantOp>( loc, returnType[0],
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

#if 0
        LLVM_DEBUG( {
            llvm::errs() << "============================\nCREATE SCOPE I:\n";
            builder.getInsertionBlock()->getParentOp()->dump();

            llvm::errs() << "Current insertion block:\n";
            builder.getInsertionBlock()->dump();

            auto insertionPoint = builder.getInsertionPoint();
            if ( insertionPoint != builder.getInsertionBlock()->end() )
            {
                llvm::errs() << "Insertion point is after operation:\n";
                ( *insertionPoint ).dump();    // Dereference the iterator to get the Operation*
            }
            else
            {
                llvm::errs() << "Insertion point is at the end of the block\n";
            }
        } );
#endif

        builder.setInsertionPoint( returnOp );    // Set insertion point *before* the saved toy::ReturnOp

#if 0
        LLVM_DEBUG( {
            llvm::errs() << "============================\nCREATE SCOPE II:\n";
            builder.getInsertionBlock()->getParentOp()->dump();

            llvm::errs() << "Current insertion block:\n";
            builder.getInsertionBlock()->dump();

            auto insertionPoint = builder.getInsertionPoint();
            if ( insertionPoint != builder.getInsertionBlock()->end() )
            {
                llvm::errs() << "Insertion point is after operation:\n";
                ( *insertionPoint ).dump();    // Dereference the iterator to get the Operation*
            }
            else
            {
                llvm::errs() << "Insertion point is at the end of the block\n";
            }
        } );
#endif

        currentFuncName = funcName;
        setFuncOp( funcOp );
    }

    void MLIRListener::mainFirstTime( mlir::Location loc )
    {
        if ( !mainScopeGenerated && (MLIRListener::currentFuncName == ENTRY_SYMBOL_NAME) )
        {
            auto funcType = builder.getFunctionType( {}, tyI32 );
            auto funcOp = builder.create<mlir::func::FuncOp>( loc, ENTRY_SYMBOL_NAME, funcType );

            std::vector<std::string> paramNames;
            createScope( loc, funcOp, ENTRY_SYMBOL_NAME, paramNames );

            mainScopeGenerated = true;
        }
    }

    void MLIRListener::enterStartRule( ToyParser::StartRuleContext *ctx )
    {
        currentFuncName = ENTRY_SYMBOL_NAME;
        auto loc = getLocation( ctx );
        setLastLoc( loc );
    }

    void MLIRListener::exitStartRule( ToyParser::StartRuleContext *ctx )
    {
        auto loc = getLocation( ctx );
        mainFirstTime( loc );

        auto lastLoc = getLastLoc( );

        if ( !wasTerminatorExplicit( ) )
        {
            processReturnLike<ToyParser::NumericLiteralContext>( lastLoc, nullptr, nullptr, nullptr );
        }
    }

    void MLIRListener::enterFunction( ToyParser::FunctionContext *ctx )
    {
        auto loc = getLocation( ctx );

        mainIP = builder.saveInsertionPoint();

        builder.setInsertionPointToStart( mod.getBody() );

        std::string funcName = ctx->IDENTIFIER()->getText();

        std::vector<mlir::Type> returns;
        if ( auto rt = ctx->scalarType() )
        {
            auto returnType = parseScalarType( rt->getText() );
            returns.push_back( returnType );
        }

        std::vector<mlir::Type> paramTypes;
        std::vector<std::string> paramNames;
        for ( auto *paramCtx : ctx->parameterTypeAndName() )
        {
            auto paramType = parseScalarType( paramCtx->scalarType()->getText() );
            auto paramName = paramCtx->IDENTIFIER()->getText();
            paramTypes.push_back( paramType );
            paramNames.push_back( paramName );

            setVarState( funcName, paramName, variable_state::assigned );
        }

        std::vector<mlir::NamedAttribute> attrs;
        attrs.push_back(
            mlir::NamedAttribute( builder.getStringAttr( "sym_visibility" ), builder.getStringAttr( "private" ) ) );

        auto funcType = builder.getFunctionType( paramTypes, returns );
        auto funcOp = builder.create<mlir::func::FuncOp>( loc, funcName, funcType, attrs );
        createScope( loc, funcOp, funcName, paramNames );
        setLastLoc( loc );
    }

    void MLIRListener::exitFunction( ToyParser::FunctionContext *ctx )
    {
        auto lastLoc = getLastLoc( );

        // This is in case the grammar enforcement of a RETURN at end of FUNCTION is removed (which would make sense, and is also
        // desirable when control flow is added.)
        // For now, still have enforced mandatory RETURN at function-end in the grammar.
        if ( !wasTerminatorExplicit( ) )
        {
            processReturnLike<ToyParser::LiteralContext>( lastLoc, nullptr, nullptr, nullptr );
        }

        builder.restoreInsertionPoint( mainIP );

        currentFuncName = ENTRY_SYMBOL_NAME;
    }

    mlir::Value MLIRListener::handleCall( ToyParser::CallContext *ctx )
    {
        auto loc = getLocation( ctx );
        auto funcName = ctx->IDENTIFIER()->getText();
        mlir::func::FuncOp funcOp = getFuncOp( funcName );
        auto funcType = funcOp.getFunctionType();
        std::vector<mlir::Value> parameters;
        if ( auto params = ctx->parameterList() )
        {
            int i = 0;

            auto psz = params->parameter().size();
            auto fsz = funcType.getInputs().size();
            assert( psz == fsz );

            for ( ToyParser::ParameterContext *p : params->parameter() )
            {
                std::string paramText = p->getText();
                std::cout << std::format( "CALL function {}: param: {}\n", funcName, paramText );

                mlir::Value value;
                auto lit = p->literal();

                std::string s;
                buildUnaryExpression( lit ? lit->BOOLEAN_PATTERN() : nullptr, lit ? lit->INTEGER_PATTERN() : nullptr,
                                      lit ? lit->FLOAT_PATTERN() : nullptr, p->IDENTIFIER(),
                                      lit ? lit->STRING_PATTERN() : nullptr, loc, value, s );

                assert( s.length() == 0 );    // for StringNode.  Want to support passing string literals (not just to
                                              // PRINT builtin), but not now.

                value = castOpIfRequired( loc, value, funcType.getInputs()[i] );
                parameters.push_back( value );
                i++;
            }
        }

        mlir::TypeRange resultTypes = funcType.getResults();

        auto callOp = builder.create<toy::CallOp>( loc, resultTypes, funcName, parameters );

        // Return the first result (or null for void calls)
        return resultTypes.empty() ? mlir::Value{} : callOp.getResults()[0];
    }

    void MLIRListener::enterCall( ToyParser::CallContext *ctx )
    {
        if ( callIsHandled )
            return;

        auto loc = getLocation( ctx );
        mainFirstTime( loc );
        setLastLoc( loc );

        handleCall( ctx );
    }

    void MLIRListener::enterDeclare( ToyParser::DeclareContext *ctx )
    {
        auto loc = getLocation( ctx );
        mainFirstTime( loc );
        setLastLoc( loc );
        auto varName = ctx->IDENTIFIER()->getText();

        registerDeclaration( loc, varName, tyF64, ctx->arrayBoundsExpression() );
    }

    void MLIRListener::enterBoolDeclare( ToyParser::BoolDeclareContext *ctx )
    {
        auto loc = getLocation( ctx );
        mainFirstTime( loc );
        setLastLoc( loc );
        auto varName = ctx->IDENTIFIER()->getText();
        registerDeclaration( loc, varName, tyI1, ctx->arrayBoundsExpression() );
    }

    void MLIRListener::enterIntDeclare( ToyParser::IntDeclareContext *ctx )
    {
        auto loc = getLocation( ctx );
        mainFirstTime( loc );
        setLastLoc( loc );
        auto varName = ctx->IDENTIFIER()->getText();

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
            throw exception_with_context( __FILE__, __LINE__, __func__,
                                          "Internal error: Unsupported signed integer declaration size.\n" );
        }
    }

    void MLIRListener::enterFloatDeclare( ToyParser::FloatDeclareContext *ctx )
    {
        auto loc = getLocation( ctx );
        mainFirstTime( loc );
        setLastLoc( loc );
        auto varName = ctx->IDENTIFIER()->getText();

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
            throw exception_with_context( __FILE__, __LINE__, __func__,
                                          "Internal error: Unsupported floating point declaration size.\n" );
        }
    }

    void MLIRListener::enterStringDeclare( ToyParser::StringDeclareContext *ctx )
    {
        auto loc = getLocation( ctx );
        mainFirstTime( loc );
        setLastLoc( loc );
        auto varName = ctx->IDENTIFIER()->getText();
        ToyParser::ArrayBoundsExpressionContext *arrayBounds = ctx->arrayBoundsExpression();
        assert( arrayBounds );

        registerDeclaration( loc, varName, tyI8, arrayBounds );
    }

    void MLIRListener::enterIfelifelse( ToyParser::IfelifelseContext *ctx )
    {
        auto loc = getLocation( ctx );
        mainFirstTime( loc );

        llvm::errs() << std::format( "{}NYI: {}\n", formatLocation( loc ), ctx->getText() );

        assert( 0 );
    }

    void MLIRListener::enterPrint( ToyParser::PrintContext *ctx )
    {
        auto loc = getLocation( ctx );
        mainFirstTime( loc );
        setLastLoc( loc );

        mlir::Type varType;

#if 0
        LLVM_DEBUG( {
            llvm::errs() << "Scope IR before toy::PrintOp creation:\n";
            builder.getInsertionBlock()->getParentOp()->dump();

            llvm::errs() << "Current insertion block:\n";
            builder.getInsertionBlock()->dump();

            auto insertionPoint = builder.getInsertionPoint();
            if ( insertionPoint != builder.getInsertionBlock()->end() )
            {
                llvm::errs() << "Insertion point is after operation:\n";
                ( *insertionPoint ).dump();    // Dereference the iterator to get the Operation*
            }
            else
            {
                llvm::errs() << "Insertion point is at the end of the block\n";
            }
        } );
#endif

        auto varNameObject = ctx->IDENTIFIER();
        if ( varNameObject )
        {
            auto varName = varNameObject->getText();
            auto varState = getVarState( varName );
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

            auto declareOp = lookupDeclareForVar( varName );

            mlir::Type elemType = declareOp.getTypeAttr().getValue();

            if ( declareOp.getSizeAttr() )    // Check if size attribute exists
            {
                // Array: load a generic pointer
                varType = tyPtr;
            }
            else
            {
                // Scalar: load the value
                varType = elemType;
            }

            auto symRef = mlir::SymbolRefAttr::get( &dialect.context, varName );
            auto value = builder.create<toy::LoadOp>( loc, varType, symRef );
            builder.create<toy::PrintOp>( loc, value );
        }
        else if ( auto theString = ctx->STRING_PATTERN() )
        {
            auto s = stripQuotes( theString->getText() );
            auto strAttr = builder.getStringAttr( s );

            auto stringLiteral = builder.create<toy::StringLiteralOp>( loc, tyPtr, strAttr );

            builder.create<toy::PrintOp>( loc, stringLiteral );
        }
        else
        {
            throw exception_with_context(
                __FILE__, __LINE__, __func__,
                std::format( "{}error: unexpected print context {}\n", formatLocation( loc ), ctx->getText() ) );
        }
    }

    // Apply type conversions to match func::FuncOp return type.  This is adapted from AssignOpLowering, but
    // uses arith dialect operations instead of LLVM dialect
    mlir::Value MLIRListener::castOpIfRequired( mlir::Location loc, mlir::Value value, mlir::Type desiredType )
    {
        if ( value.getType() != desiredType )
        {
            auto valType = value.getType();

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
            else if ( auto viType = mlir::cast<mlir::IntegerType>( valType ) )
            {
                auto vwidth = viType.getWidth();
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
                else if ( auto miType = mlir::cast<mlir::IntegerType>( desiredType ) )
                {
                    // boolr: mwidth == 32, vwidth == 1
                    auto mwidth = miType.getWidth();
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
    void MLIRListener::processReturnLike( mlir::Location loc, Literal *lit, tNode *var, tNode *boolNode )
    {
        // Handle the dummy ReturnOp originally inserted in the FuncOp's block
        auto *currentBlock = builder.getInsertionBlock();
        assert( !currentBlock->empty() );
        auto *parentOp = currentBlock->getParentOp();
        if ( !isa<toy::ScopeOp>( parentOp ) )
        {
            throw exception_with_context(
                __FILE__, __LINE__, __func__,
                std::format( "{}error: RETURN statement must be inside a toy.scope\n", formatLocation( loc ) ) );
        }

        auto func = parentOp->getParentOfType<mlir::func::FuncOp>();
        auto returnType = func.getFunctionType().getResults();

        assert( isa<toy::ReturnOp>( currentBlock->getTerminator() ) );

        // Erase existing toy::ReturnOp and its constant
        mlir::Operation *existingExit = currentBlock->getTerminator();
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

        // always regenerate the RETURN/EXIT so that we have the terminator location set properly (not the function body start location
        // that was used to create the dummy toy.ReturnOp that we rewrite here.)
        if ( lit || var || boolNode )
        {
            std::string s;
            buildUnaryExpression( boolNode, lit ? lit->INTEGER_PATTERN() : nullptr, lit ? lit->FLOAT_PATTERN() : nullptr,
                                  var,
                                  nullptr,    // stringNode
                                  loc, value, s );
            assert( s.length() == 0 );

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
                // FIXME: factor this out into a generateZeroMatchingType() function.
                if ( auto intType = mlir::dyn_cast<mlir::IntegerType>( returnType[0] ) )
                {
                    auto width = intType.getWidth();
                    value = builder.create<mlir::arith::ConstantIntOp>( loc, 0, width );
                }
                else if ( auto floatType = mlir::dyn_cast<mlir::FloatType>( returnType[0] ) )
                {
                    llvm::APFloat apVal( 0.0 );
                    if ( floatType.getWidth() == 64 )
                    {
                        value = builder.create<mlir::arith::ConstantFloatOp>( loc, apVal, tyF64 );
                    }
                    else if ( floatType.getWidth() == 32 )
                    {
                        value = builder.create<mlir::arith::ConstantFloatOp>( loc, apVal, tyF32 );
                    }
                    else
                    {
                        assert( 0 );
                    }
                }
                else
                {
                    assert( 0 );
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
            builder.create<toy::ReturnOp>( loc, mlir::ValueRange{ } );
        }

        markExplicitTerminator();
    }

    void MLIRListener::enterReturnStatement( ToyParser::ReturnStatementContext *ctx )
    {
        auto loc = getLocation( ctx );
        mainFirstTime( loc );
        setLastLoc( loc );

        auto lit = ctx->literal();
        auto var = ctx->IDENTIFIER();

        processReturnLike<ToyParser::LiteralContext>( loc, lit, var, lit ? lit->BOOLEAN_PATTERN() : nullptr );
    }

    void MLIRListener::enterExitStatement( ToyParser::ExitStatementContext *ctx )
    {
        auto loc = getLocation( ctx );
        mainFirstTime( loc );
        setLastLoc( loc );

        auto lit = ctx->numericLiteral();
        auto var = ctx->IDENTIFIER();

        processReturnLike<ToyParser::NumericLiteralContext>( loc, lit, var, nullptr );
    }

    void MLIRListener::enterAssignment( ToyParser::AssignmentContext *ctx )
    {
        assignmentTargetValid = true;
        auto loc = getLocation( ctx );
        mainFirstTime( loc );
        setLastLoc( loc );
        currentVarName = ctx->IDENTIFIER()->getText();
        auto varState = getVarState( currentVarName );
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
        callIsHandled = false;
    }

    void MLIRListener::enterAssignmentExpression( ToyParser::AssignmentExpressionContext *ctx )
    {
        if ( !assignmentTargetValid )
        {
            return;
        }
        auto loc = getLocation( ctx );
        mainFirstTime( loc );
        setLastLoc( loc );
        mlir::Value resultValue;

        auto declareOp = lookupDeclareForVar( currentVarName );
        mlir::TypeAttr typeAttr = declareOp.getTypeAttr();
        mlir::Type opType = typeAttr.getValue();

        mlir::Value lhsValue;
        auto bsz = ctx->binaryElement().size();
        std::string s;

        if ( bsz == 0 )
        {
            mlir::Value lhsValue;

            auto lit = ctx->literal();

            if ( auto call = ctx->call() )
            {
                callIsHandled = true;
                lhsValue = handleCall( call );
            }
            else
            {
                buildUnaryExpression( lit ? lit->BOOLEAN_PATTERN() : nullptr, lit ? lit->INTEGER_PATTERN() : nullptr,
                                      lit ? lit->FLOAT_PATTERN() : nullptr, ctx->IDENTIFIER(),
                                      lit ? lit->STRING_PATTERN() : nullptr, loc, lhsValue, s );
            }

            resultValue = lhsValue;
            if ( auto unaryOp = ctx->unaryOperator() )
            {
                auto opText = unaryOp->getText();
                if ( opText == "-" )
                {
                    auto op = builder.create<toy::NegOp>( loc, opType, lhsValue );
                    resultValue = op.getResult();
                    assert( s.length() == 0 );
                }
                else if ( opText == "NOT" )
                {
                    auto rhsValue = builder.create<mlir::arith::ConstantIntOp>( loc, 0, 64 );

                    auto b = builder.create<toy::EqualOp>( loc, opType, lhsValue, rhsValue );
                    resultValue = b.getResult();
                    assert( s.length() == 0 );
                }
            }
        }
        else
        {
            assert( bsz == 2 );

            auto lhs = ctx->binaryElement()[0];
            auto rhs = ctx->binaryElement()[1];
            auto opText = ctx->binaryOperator()->getText();

            auto llit = lhs->numericLiteral();
            buildUnaryExpression( nullptr,    // booleanNode
                                  llit ? llit->INTEGER_PATTERN() : nullptr, llit ? llit->FLOAT_PATTERN() : nullptr,
                                  lhs->IDENTIFIER(),
                                  nullptr,    // stringNode
                                  loc, lhsValue, s );
            assert( s.length() == 0 );

            mlir::Value rhsValue;
            auto rlit = rhs->numericLiteral();
            buildUnaryExpression( nullptr,    // booleanNode
                                  rlit ? rlit->INTEGER_PATTERN() : nullptr, rlit ? rlit->FLOAT_PATTERN() : nullptr,
                                  rhs->IDENTIFIER(),
                                  nullptr,    // stringNode
                                  loc, rhsValue, s );
            assert( s.length() == 0 );

            // Create the binary operator (supports +, -, *, /)
            if ( opText == "+" )
            {
                auto b = builder.create<toy::AddOp>( loc, opType, lhsValue, rhsValue );
                resultValue = b.getResult();
            }
            else if ( opText == "-" )
            {
                auto b = builder.create<toy::SubOp>( loc, opType, lhsValue, rhsValue );
                resultValue = b.getResult();
            }
            else if ( opText == "*" )
            {
                auto b = builder.create<toy::MulOp>( loc, opType, lhsValue, rhsValue );
                resultValue = b.getResult();
            }
            else if ( opText == "/" )
            {
                auto b = builder.create<toy::DivOp>( loc, opType, lhsValue, rhsValue );
                resultValue = b.getResult();
            }
            else if ( opText == "<" )
            {
                auto b = builder.create<toy::LessOp>( loc, opType, lhsValue, rhsValue );
                resultValue = b.getResult();
            }
            else if ( opText == ">" )
            {
                auto b = builder.create<toy::LessOp>( loc, opType, rhsValue, lhsValue );
                resultValue = b.getResult();
            }
            else if ( opText == "<=" )
            {
                auto b = builder.create<toy::LessEqualOp>( loc, opType, lhsValue, rhsValue );
                resultValue = b.getResult();
            }
            else if ( opText == ">=" )
            {
                auto b = builder.create<toy::LessEqualOp>( loc, opType, rhsValue, lhsValue );
                resultValue = b.getResult();
            }
            else if ( opText == "EQ" )
            {
                auto b = builder.create<toy::EqualOp>( loc, opType, lhsValue, rhsValue );
                resultValue = b.getResult();
            }
            else if ( opText == "NE" )
            {
                auto b = builder.create<toy::NotEqualOp>( loc, opType, lhsValue, rhsValue );
                resultValue = b.getResult();
            }
            else if ( opText == "AND" )
            {
                auto b = builder.create<toy::AndOp>( loc, opType, lhsValue, rhsValue );
                resultValue = b.getResult();
            }
            else if ( opText == "OR" )
            {
                auto b = builder.create<toy::OrOp>( loc, opType, lhsValue, rhsValue );
                resultValue = b.getResult();
            }
            else if ( opText == "XOR" )
            {
                auto b = builder.create<toy::XorOp>( loc, opType, lhsValue, rhsValue );
                resultValue = b.getResult();
            }
            else
            {
                throw exception_with_context( __FILE__, __LINE__, __func__,
                                              std::format( "error: Invalid binary operator {}\n", opText ) );
            }
        }

        assert( !currentVarName.empty() );

        auto symRef = mlir::SymbolRefAttr::get( &dialect.context, currentVarName );
        if ( s.length() )
        {
            auto strAttr = builder.getStringAttr( s );

            auto stringLiteral = builder.create<toy::StringLiteralOp>( loc, tyPtr, strAttr );

            builder.create<toy::AssignOp>( loc, symRef, stringLiteral );
        }
        else
        {
            builder.create<toy::AssignOp>( loc, symRef, resultValue );
        }

        setVarState( currentFuncName, currentVarName, variable_state::assigned );
        currentVarName.clear();
    }
}    // namespace toy

// vim: et ts=4 sw=4
