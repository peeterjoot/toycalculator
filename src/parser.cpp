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
        auto parentFunc = funcByName[currentFuncName];
        auto funcOp = mlir::dyn_cast<toy::FuncOp>( parentFunc );
        LLVM_DEBUG( {
            llvm::errs() << std::format( "Lookup symbol {} in parent function:\n", varName );
            funcOp->dump();
        } );

        auto *symbolOp = mlir::SymbolTable::lookupSymbolIn( funcOp, varName );
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
    inline std::string MLIRListener::buildUnaryExpression( tNode *booleanNode, tNode *integerNode, tNode *floatNode,
                                                           tNode *variableNode, tNode *stringNode, mlir::Location loc,
                                                           mlir::Value &value )
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
            value = builder.create<mlir::arith::ConstantFloatOp>( loc, apVal, builder.getF64Type() );
        }
        else if ( variableNode )
        {
            auto varName = variableNode->getText();

            auto varState = getVarState( currentFuncName, varName );

            if ( varState == variable_state::undeclared )
            {
                lastSemError = semantic_errors::variable_not_declared;
                throw exception_with_context(
                    __FILE__, __LINE__, __func__,
                    std::format( "{}error: Variable {} not declared in expr\n", formatLocation( loc ), varName ) );
            }

            if ( varState != variable_state::assigned )
            {
                lastSemError = semantic_errors::variable_not_assigned;
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
            return stripQuotes( stringNode->getText() );
        }
        else
        {
            throw exception_with_context( __FILE__, __LINE__, __func__,
                                          std::format( "{}error: Invalid operand\n", formatLocation( loc ) ) );
        }

        return std::string();
    }

    // \retval true if error
    inline bool MLIRListener::registerDeclaration( mlir::Location loc, const std::string &varName, mlir::Type ty,
                                                   ToyParser::ArrayBoundsExpressionContext *arrayBounds )
    {
        auto varState = getVarState( currentFuncName, varName );
        if ( varState != variable_state::undeclared )
        {
            lastSemError = semantic_errors::variable_already_declared;
            llvm::errs() << std::format( "{}error: Variable {} already declared\n", formatLocation( loc ), varName );
            return true;
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
            auto dcl = builder.create<toy::DeclareOp>( loc, mlir::TypeAttr::get( ty ), sizeAttr );
            dcl->setAttr( "sym_name", strAttr );
        }
        else
        {
            auto dcl = builder.create<toy::DeclareOp>( loc, mlir::TypeAttr::get( ty ), nullptr );
            dcl->setAttr( "sym_name", strAttr );
        }

        // For test purposes to verify that symbol lookup for varName worked right after the DeclareOp build call:
        // auto ddcl = lookupDeclareForVar( varName );

        return false;
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

    void MLIRListener::enterStartRule( ToyParser::StartRuleContext *ctx )
    {
        auto loc = getLocation( ctx );

        auto funcType = builder.getFunctionType( {}, tyI32 );
        std::vector<mlir::Attribute> paramAttrs;
        auto func =
            builder.create<toy::FuncOp>( loc, std::string( ENTRY_SYMBOL_NAME ), funcType,
                                         builder.getArrayAttr( paramAttrs ), builder.getStringAttr( "public" ) );

        currentFuncName = ENTRY_SYMBOL_NAME;
        funcByName[currentFuncName] = func;
        auto &block = *func.addEntryBlock();
        builder.setInsertionPointToStart( &block );
    }

    void MLIRListener::exitStartRule( ToyParser::StartRuleContext *ctx )
    {
        if ( lastOp != lastOperator::exitOp )
        {
            // Empty ValueRange means we are building a toy.return with no
            // operands:
            builder.create<toy::ExitOp>( lastLocation, mlir::ValueRange{} );
        }

        builder.setInsertionPointToEnd( mod.getBody() );
        if ( lastSemError != semantic_errors::not_an_error )
        {
            // don't care what the error was, since we already logged it to the
            // console.  Just throw, avoiding future LLVM IR lowering:
            throw exception_with_context( __FILE__, __LINE__, __func__, "semantic exception" );
        }
    }

    void MLIRListener::enterFunction( ToyParser::FunctionContext *ctx )
    {
        auto loc = getLocation( ctx );

        mainIP = builder.saveInsertionPoint();

        builder.setInsertionPointToStart( mod.getBody() );

        std::vector<mlir::Type> returns;
        if ( auto rt = ctx->scalarType() )
        {
            auto returnType = parseScalarType( rt->getText() );
            returns.push_back( returnType );
        }

        std::string funcName = ctx->IDENTIFIER()->getText();

        // Collect parameter types and names
        std::vector<mlir::Type> paramTypes;
        std::vector<mlir::Attribute> paramAttrs;
        for ( auto *paramCtx : ctx->parameterTypeAndName() )
        {
            auto paramType = parseScalarType( paramCtx->scalarType()->getText() );
            auto paramName = paramCtx->IDENTIFIER()->getText();
            paramTypes.push_back( paramType );
            paramAttrs.push_back( mlir::DictionaryAttr::get( builder.getContext(),
                                                             { { "sym_name", builder.getStringAttr( paramName ) } } ) );
        }

        auto funcType = builder.getFunctionType( paramTypes, returns );
        auto func = builder.create<toy::FuncOp>( loc, funcName, funcType, builder.getArrayAttr( paramAttrs ),
                                                 builder.getStringAttr( "private" ) );
        auto &block = *func.addEntryBlock();
        builder.setInsertionPointToStart( &block );

        currentFuncName = funcName;
        funcByName[currentFuncName] = func;
    }

    void MLIRListener::exitFunction( ToyParser::FunctionContext *ctx )
    {
        // Could add the return if it wasn't done, as done for exit.  Instead, perhaps temporarily (at
        // least until ready to support premature return, when control flow possibilities are allowed),
        // have enforced mandatory RETURN at function end in the grammar.
        currentFuncName = ENTRY_SYMBOL_NAME;

        builder.restoreInsertionPoint( mainIP );
    }

    void MLIRListener::enterCall( ToyParser::CallContext *ctx)
    {
        //std::cout << ctx->getText() << '\n';
        lastOp = lastOperator::callOp;
        auto loc = getLocation( ctx );

        auto function = ctx->IDENTIFIER()->getText();
        std::vector<mlir::Operation*> parameters;
        if ( auto params = ctx->parameterList() )
        {
            for ( ToyParser::ParameterContext * p : params->parameter() )
            {
                std::cout << std::format( "param: {}\n", p->getText() );
                assert( 0 );
            }
        }

        auto op = funcByName[function];
        auto funcOp = mlir::dyn_cast<toy::FuncOp>( op );
        auto funcType = funcOp.getFunctionTypeAttrValue();
        auto resultType = funcType.getNumResults() ? funcType.getResults()[0] : tyVoid;

        builder.create<mlir::func::CallOp>( loc, function, mlir::TypeRange{ resultType }, mlir::ValueRange{} );
    //builder.create<mlir::func::CallOp>( printLoc, "__toy_print_i64", mlir::TypeRange{}, mlir::ValueRange{ val } );
    }

    void MLIRListener::enterDeclare( ToyParser::DeclareContext *ctx )
    {
        lastOp = lastOperator::declareOp;
        auto loc = getLocation( ctx );
        auto varName = ctx->IDENTIFIER()->getText();

        registerDeclaration( loc, varName, tyF64, ctx->arrayBoundsExpression() );
    }

    void MLIRListener::enterBoolDeclare( ToyParser::BoolDeclareContext *ctx )
    {
        lastOp = lastOperator::declareOp;
        auto loc = getLocation( ctx );
        auto varName = ctx->IDENTIFIER()->getText();
        registerDeclaration( loc, varName, tyI1, ctx->arrayBoundsExpression() );
    }

    void MLIRListener::enterIntDeclare( ToyParser::IntDeclareContext *ctx )
    {
        lastOp = lastOperator::declareOp;
        auto loc = getLocation( ctx );
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
        lastOp = lastOperator::declareOp;
        auto loc = getLocation( ctx );
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
        lastOp = lastOperator::declareOp;
        auto loc = getLocation( ctx );
        auto varName = ctx->IDENTIFIER()->getText();
        ToyParser::ArrayBoundsExpressionContext *arrayBounds = ctx->arrayBoundsExpression();
        assert( arrayBounds );

        registerDeclaration( loc, varName, tyI8, arrayBounds );
    }

    void MLIRListener::enterIfelifelse( ToyParser::IfelifelseContext *ctx )
    {
        lastOp = lastOperator::ifOp;
        auto loc = getLocation( ctx );

        llvm::errs() << std::format( "{}NYI: {}\n", formatLocation( loc ), ctx->getText() );

        assert( 0 );
    }

    void MLIRListener::enterPrint( ToyParser::PrintContext *ctx )
    {
        lastOp = lastOperator::printOp;
        auto loc = getLocation( ctx );

        mlir::Type varType;

        auto varNameObject = ctx->IDENTIFIER();
        if ( varNameObject )
        {
            auto varName = varNameObject->getText();
            auto varState = getVarState( currentFuncName, varName );
            if ( varState == variable_state::undeclared )
            {
                lastSemError = semantic_errors::variable_not_declared;
                llvm::errs() << std::format( "{}error: Variable {} not declared in PRINT\n", formatLocation( loc ),
                                             varName );
                return;
            }
            if ( varState != variable_state::assigned )
            {
                lastSemError = semantic_errors::variable_not_assigned;
                llvm::errs() << std::format( "{}error: Variable {} not assigned in PRINT\n", formatLocation( loc ),
                                             varName );
                return;
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

    template <class Literal>
    void MLIRListener::processReturnLike( mlir::Location loc, Literal *lit, tNode *var, tNode *boolNode )
    {
        if ( ( lit == nullptr ) && ( var == nullptr ) )
        {
            //  Create toy.return with no operands (empty ValueRange)
            builder.create<toy::ExitOp>( loc, mlir::ValueRange{} );
        }
        else
        {
            mlir::Value value;

            auto s = buildUnaryExpression( boolNode, lit ? lit->INTEGER_PATTERN() : nullptr,
                                           lit ? lit->FLOAT_PATTERN() : nullptr, var,
                                           nullptr,    // stringNode
                                           loc, value );
            assert( s.length() == 0 );

            builder.create<toy::ExitOp>( loc, mlir::ValueRange{ value } );
        }
    }

    void MLIRListener::enterReturnStatement( ToyParser::ReturnStatementContext *ctx )
    {
        lastOp = lastOperator::returnOp;
        auto loc = getLocation( ctx );

        auto lit = ctx->literal();
        auto var = ctx->IDENTIFIER();

        processReturnLike<ToyParser::LiteralContext>( loc, lit, var, lit ? lit->BOOLEAN_PATTERN() : nullptr );
    }

    void MLIRListener::enterExitStatement( ToyParser::ExitStatementContext *ctx )
    {
        lastOp = lastOperator::exitOp;
        auto loc = getLocation( ctx );

        auto lit = ctx->numericLiteral();
        auto var = ctx->IDENTIFIER();

        processReturnLike<ToyParser::NumericLiteralContext>( loc, lit, var, nullptr );
    }

    void MLIRListener::enterAssignment( ToyParser::AssignmentContext *ctx )
    {
        lastOp = lastOperator::assignmentOp;
        assignmentTargetValid = true;
        auto loc = getLocation( ctx );
        currentVarName = ctx->IDENTIFIER()->getText();
        auto varState = getVarState( currentFuncName, currentVarName );
        if ( varState == variable_state::declared )
        {
            setVarState( currentFuncName, currentVarName, variable_state::assigned );
        }
        else if ( varState == variable_state::undeclared )
        {
            lastSemError = semantic_errors::variable_not_declared;
            llvm::errs() << std::format( "{}error: Variable {} not declared in assignment\n", formatLocation( loc ),
                                         currentVarName );
            assignmentTargetValid = false;
        }
        currentAssignLoc = loc;
    }

    void MLIRListener::enterAssignmentExpression( ToyParser::AssignmentExpressionContext *ctx )
    {
        if ( !assignmentTargetValid )
        {
            return;
        }
        auto loc = getLocation( ctx );
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

            s = buildUnaryExpression( lit ? lit->BOOLEAN_PATTERN() : nullptr, lit ? lit->INTEGER_PATTERN() : nullptr,
                                      lit ? lit->FLOAT_PATTERN() : nullptr, ctx->IDENTIFIER(),
                                      lit ? lit->STRING_PATTERN() : nullptr, loc, lhsValue );

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
            s = buildUnaryExpression( nullptr,    // booleanNode
                                      llit ? llit->INTEGER_PATTERN() : nullptr, llit ? llit->FLOAT_PATTERN() : nullptr,
                                      lhs->IDENTIFIER(),
                                      nullptr,    // stringNode
                                      loc, lhsValue );
            assert( s.length() == 0 );

            mlir::Value rhsValue;
            auto rlit = rhs->numericLiteral();
            s = buildUnaryExpression( nullptr,    // booleanNode
                                      rlit ? rlit->INTEGER_PATTERN() : nullptr, rlit ? rlit->FLOAT_PATTERN() : nullptr,
                                      rhs->IDENTIFIER(),
                                      nullptr,    // stringNode
                                      loc, rhsValue );
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
