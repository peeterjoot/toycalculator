/**
 * @file    parser.cpp
 * @author  Peeter Joot <peeterjoot@pm.me>
 * @brief   altlr4 parse tree listener and MLIR builder.
 */
#include <llvm/Support/Debug.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Types.h>

#include <format>

#include "ToyExceptions.h"
#include "parser.h"

#define DEBUG_TYPE "toy-parser"

namespace toy
{
    DialectCtx::DialectCtx()
    {
        context.getOrLoadDialect<toy::ToyDialect>();
        context.getOrLoadDialect<mlir::arith::ArithDialect>();
        context.getOrLoadDialect<mlir::memref::MemRefDialect>();
        context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
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

    // \retval true if error
    inline std::string MLIRListener::buildUnaryExpression( tNode *booleanNode, tNode *integerNode, tNode *floatNode,
                                                           tNode *variableNode, tNode *stringNode, mlir::Location loc,
                                                           mlir::Value &value, theTypes &ty )
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
            ty = theTypes::boolean;
        }
        else if ( integerNode )
        {
            int64_t val = std::stoll( integerNode->getText() );
            value = builder.create<mlir::arith::ConstantIntOp>( loc, val, 64 );
            ty = theTypes::integer64;
        }
        else if ( floatNode )
        {
            double val = std::stod( floatNode->getText() );

            llvm::APFloat apVal( val );

            // Like the INTEGER_PATTERN node above, create the float literal with
            // the max sized type. Would need a grammar change to have a
            // specific type (i.e.: size) associated with literals.
            value = builder.create<mlir::arith::ConstantFloatOp>( loc, apVal, builder.getF64Type() );
            ty = theTypes::float64;
        }
        else if ( variableNode )
        {
            auto varName = variableNode->getText();

            auto varState = var_states[varName];

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

            auto dcl = var_storage[varName];
            auto declareOp = mlir::dyn_cast<toy::DeclareOp>( dcl );

            mlir::Type varType = declareOp.getTypeAttr().getValue();
            value = builder.create<toy::LoadOp>( loc, varType, builder.getStringAttr( varName ) );

            ty = getCompilerType( varType );
        }
        else if ( stringNode )
        {
            ty = theTypes::string;

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
        auto varState = var_states[varName];
        if ( varState != variable_state::undeclared )
        {
            lastSemError = semantic_errors::variable_already_declared;
            llvm::errs() << std::format( "{}error: Variable {} already declared\n", formatLocation( loc ), varName );
            return true;
        }

        var_states[varName] = variable_state::declared;

        size_t arraySize{};
        if ( arrayBounds )
        {
            auto index = arrayBounds->INTEGER_PATTERN();
            arraySize = std::stoi( index->getText() );
        }

        if ( arraySize )
        {
            auto sizeAttr = builder.getI64IntegerAttr( arraySize );
            auto dcl = builder.create<toy::DeclareOp>( loc, builder.getStringAttr( varName ), mlir::TypeAttr::get( ty ),
                                                       sizeAttr );
            var_storage[varName] = dcl;
        }
        else
        {
            auto dcl = builder.create<toy::DeclareOp>( loc, builder.getStringAttr( varName ), mlir::TypeAttr::get( ty ),
                                                       nullptr );
            var_storage[varName] = dcl;
        }

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
    }

    void MLIRListener::enterStartRule( ToyParser::StartRuleContext *ctx )
    {
        auto loc = getLocation( ctx );
        programOp = builder.create<toy::ProgramOp>( loc );
        programOp.getBody().push_back( new mlir::Block() );
        builder.setInsertionPointToStart( &programOp.getBody().front() );
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

    void MLIRListener::enterDeclare( ToyParser::DeclareContext *ctx )
    {
        lastOp = lastOperator::declareOp;
        auto loc = getLocation( ctx );
        auto varName = ctx->VARIABLENAME_PATTERN()->getText();

        registerDeclaration( loc, varName, builder.getF64Type(), ctx->arrayBoundsExpression() );
    }

    void MLIRListener::enterBoolDeclare( ToyParser::BoolDeclareContext *ctx )
    {
        lastOp = lastOperator::declareOp;
        auto loc = getLocation( ctx );
        auto varName = ctx->VARIABLENAME_PATTERN()->getText();
        registerDeclaration( loc, varName, builder.getI1Type(), ctx->arrayBoundsExpression() );
    }

    void MLIRListener::enterIntDeclare( ToyParser::IntDeclareContext *ctx )
    {
        lastOp = lastOperator::declareOp;
        auto loc = getLocation( ctx );
        auto varName = ctx->VARIABLENAME_PATTERN()->getText();

        if ( ctx->INT8_TOKEN() )
        {
            registerDeclaration( loc, varName, builder.getI8Type(), ctx->arrayBoundsExpression() );
        }
        else if ( ctx->INT16_TOKEN() )
        {
            registerDeclaration( loc, varName, builder.getI16Type(), ctx->arrayBoundsExpression() );
        }
        else if ( ctx->INT32_TOKEN() )
        {
            registerDeclaration( loc, varName, builder.getI32Type(), ctx->arrayBoundsExpression() );
        }
        else if ( ctx->INT64_TOKEN() )
        {
            registerDeclaration( loc, varName, builder.getI64Type(), ctx->arrayBoundsExpression() );
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
        auto varName = ctx->VARIABLENAME_PATTERN()->getText();

        if ( ctx->FLOAT32_TOKEN() )
        {
            registerDeclaration( loc, varName, builder.getF32Type(), ctx->arrayBoundsExpression() );
        }
        else if ( ctx->FLOAT64_TOKEN() )
        {
            registerDeclaration( loc, varName, builder.getF64Type(), ctx->arrayBoundsExpression() );
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
        auto varName = ctx->VARIABLENAME_PATTERN()->getText();
        ToyParser::ArrayBoundsExpressionContext *arrayBounds = ctx->arrayBoundsExpression();
        assert( arrayBounds );

        registerDeclaration( loc, varName, builder.getI8Type(), arrayBounds );
    }

    void MLIRListener::enterPrint( ToyParser::PrintContext *ctx )
    {
        lastOp = lastOperator::printOp;
        auto loc = getLocation( ctx );
        auto varName = ctx->VARIABLENAME_PATTERN()->getText();
        auto varState = var_states[varName];
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

        auto dcl = var_storage[varName];
        auto declareOp = mlir::dyn_cast<toy::DeclareOp>( dcl );

        mlir::Type varType;
        mlir::Type elemType = declareOp.getTypeAttr().getValue();

        if ( declareOp.getSizeAttr() )    // Check if size attribute exists
        {
            // Array: load a generic pointer
            varType = mlir::LLVM::LLVMPointerType::get(builder.getContext(), /*addressSpace=*/0);
        }
        else
        {
            // Scalar: load the value
            varType = elemType;
        }

        auto value = builder.create<toy::LoadOp>( loc, varType, builder.getStringAttr( varName ) );

        builder.create<toy::PrintOp>( loc, value );
    }

    void MLIRListener::enterExitStatement( ToyParser::ExitStatementContext *ctx )
    {
        lastOp = lastOperator::exitOp;
        auto loc = getLocation( ctx );

        auto lit = ctx->numericLiteral();
        auto var = ctx->VARIABLENAME_PATTERN();

        if ( ( lit == nullptr ) && ( var == nullptr ) )
        {
            //  Create toy.return with no operands (empty ValueRange)
            builder.create<toy::ExitOp>( loc, mlir::ValueRange{} );
        }
        else
        {
            mlir::Value value;

            theTypes ty;
            auto s =
                buildUnaryExpression( nullptr,    // booleanNode
                                      lit ? lit->INTEGER_PATTERN() : nullptr, lit ? lit->FLOAT_PATTERN() : nullptr, var,
                                      nullptr,    // stringNode
                                      loc, value, ty );
            assert( s.length() == 0 );

            builder.create<toy::ExitOp>( loc, mlir::ValueRange{ value } );
        }
    }

    void MLIRListener::enterAssignment( ToyParser::AssignmentContext *ctx )
    {
        lastOp = lastOperator::assignmentOp;
        assignmentTargetValid = true;
        auto loc = getLocation( ctx );
        currentVarName = ctx->VARIABLENAME_PATTERN()->getText();
        auto varState = var_states[currentVarName];
        if ( varState == variable_state::declared )
        {
            var_states[currentVarName] = variable_state::assigned;
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

        auto dcl = var_storage[currentVarName];
        auto declareOp = mlir::dyn_cast<toy::DeclareOp>( dcl );
        mlir::TypeAttr typeAttr = declareOp.getTypeAttr();
        mlir::Type opType = typeAttr.getValue();

        mlir::Value lhsValue;
        theTypes lty;
        auto bsz = ctx->binaryElement().size();
        std::string s;

        if ( bsz == 0 )
        {
            mlir::Value lhsValue;

            auto lit = ctx->literal();

            theTypes ty;
            s = buildUnaryExpression( lit ? lit->BOOLEAN_PATTERN() : nullptr, lit ? lit->INTEGER_PATTERN() : nullptr,
                                      lit ? lit->FLOAT_PATTERN() : nullptr, ctx->VARIABLENAME_PATTERN(),
                                      lit ? lit->STRING_PATTERN() : nullptr, loc, lhsValue, ty );

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
                                      lhs->VARIABLENAME_PATTERN(),
                                      nullptr,    // stringNode
                                      loc, lhsValue, lty );
            assert( s.length() == 0 );

            mlir::Value rhsValue;
            theTypes rty;
            auto rlit = rhs->numericLiteral();
            s = buildUnaryExpression( nullptr,    // booleanNode
                                      rlit ? rlit->INTEGER_PATTERN() : nullptr, rlit ? rlit->FLOAT_PATTERN() : nullptr,
                                      rhs->VARIABLENAME_PATTERN(),
                                      nullptr,    // stringNode
                                      loc, rhsValue, rty );
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

        if ( s.length() )
        {
            auto strAttr = builder.getStringAttr( s );
            builder.create<toy::AssignStringOp>( loc, builder.getStringAttr( currentVarName ), strAttr );
        }
        else
        {
            builder.create<toy::AssignOp>( loc, builder.getStringAttr( currentVarName ), resultValue );
        }

        var_states[currentVarName] = variable_state::assigned;
        currentVarName.clear();
    }
}    // namespace toy

// vim: et ts=4 sw=4
