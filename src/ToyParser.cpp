/**
 * @file    ToyParser.cpp
 * @author  Peeter Joot <peeterjoot@pm.me>
 * @brief   altlr4 parse tree listener and MLIR builder.
 */
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Types.h>

#include <format>

#include "ToyExceptions.h"
#include "ToyParser.h"

namespace toy
{
    DialectCtx::DialectCtx()
    {
        context.getOrLoadDialect<toy::ToyDialect>();
        context.getOrLoadDialect<mlir::arith::ArithDialect>();
        context.getOrLoadDialect<mlir::memref::MemRefDialect>();
    }

    inline mlir::Location MLIRListener::getLocation(
        antlr4::ParserRuleContext *ctx )
    {
        size_t line = 1;
        size_t col = 0;
        if ( ctx )
        {
            line = ctx->getStart()->getLine();
            col = ctx->getStart()->getCharPositionInLine();
        }

        return mlir::FileLineColLoc::get( builder.getStringAttr( filename ),
                                          line, col + 1 );
    }

    inline std::string MLIRListener::formatLocation( mlir::Location loc )
    {
        if ( auto fileLoc = mlir::dyn_cast<mlir::FileLineColLoc>( loc ) )
        {
            return std::format( "{}:{}:{}: ", fileLoc.getFilename().str(),
                                fileLoc.getLine(), fileLoc.getColumn() );
        }
        return "";
    }

    // \retval true if error
    inline bool MLIRListener::buildUnaryExpression(
        tNode *booleanNode, tNode *integerNode, tNode *floatNode,
        tNode *variableNode, mlir::Location loc, mlir::Value &value )
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
                llvm::errs() << std::format(
                    "{}error: Internal error: boolean value neither TRUE nor "
                    "FALSE.\n",
                    formatLocation( loc ) );
                throw semantic_exception();
            }

            value = builder.create<mlir::arith::ConstantIntOp>( loc, val, 1 );
        }
        else if ( integerNode )
        {
            int64_t val = std::stoi( integerNode->getText() );
            value = builder.create<mlir::arith::ConstantIntOp>( loc, val, 64 );

#if 0
            if ( asFloat )
            {
                // Niavely assuming that this integer value can be represented
                // in a double:
                llvm::APFloat apVal( llvm::APFloat::IEEEdouble(), val );

                value = builder.create<mlir::arith::ConstantFloatOp>(
                    loc, apVal, builder.getF64Type() );
            }
#endif
        }
        else if ( floatNode )
        {
            double val = std::stod( floatNode->getText() );

            llvm::APFloat apVal( val );

            value = builder.create<mlir::arith::ConstantFloatOp>(
                loc, apVal, builder.getF64Type() );
        }
        else if ( variableNode )
        {
            auto varName = variableNode->getText();

            auto varState = var_states[varName];

            if ( varState == variable_state::undeclared )
            {
                lastSemError = semantic_errors::variable_not_declared;
                llvm::errs() << std::format(
                    "{}error: Variable {} not declared in expr\n",
                    formatLocation( loc ), varName );
                return true;
            }

            if ( varState != variable_state::assigned )
            {
                lastSemError = semantic_errors::variable_not_assigned;
                llvm::errs() << std::format(
                    "{}error: Variable {} not assigned in expr\n",
                    formatLocation( loc ), varName );
                return true;
            }

            // Load from memref<f64>
            auto memref = var_storage[varName];
            value = builder.create<mlir::memref::LoadOp>( loc, memref );
        }
        else
        {
            lastSemError = semantic_errors::unknown_error;
            llvm::errs() << std::format( "{}error: Invalid operand\n",
                                         formatLocation( loc ) );
            return true;
        }

        return false;
    }

    // \retval true if error
    inline bool MLIRListener::registerDeclaration( mlir::Location loc,
                                                   const std::string &varName )
    {
        auto varState = var_states[varName];
        if ( varState != variable_state::undeclared )
        {
            lastSemError = semantic_errors::variable_already_declared;
            llvm::errs() << std::format(
                "{}error: Variable {} already declared\n",
                formatLocation( loc ), varName );
            return true;
        }

        var_states[varName] = variable_state::declared;
        return false;
    }

    MLIRListener::MLIRListener( const std::string &_filename )
        : filename( _filename ),
          dialect(),
          builder( &dialect.context ),
          mod( mlir::ModuleOp::create( getLocation( nullptr ) ) ),
          currentAssignLoc( getLocation( nullptr ) )
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
        if ( lastOp != lastOperator::returnOp )
        {
            auto loc = getLocation( ctx );
            // Empty ValueRange means we are building a toy.return with no
            // operands:
            builder.create<toy::ExitOp>( loc, mlir::ValueRange{} );
        }

        builder.setInsertionPointToEnd( mod.getBody() );
        if ( lastSemError != semantic_errors::not_an_error )
        {
            // don't care what the error was, since we already logged it to the
            // console.  Just throw, avoiding future LLVM IR lowering:
            throw semantic_exception();
        }
    }

    void MLIRListener::enterDeclare( ToyParser::DeclareContext *ctx )
    {
        lastOp = lastOperator::declareOp;
        auto loc = getLocation( ctx );
        auto varName = ctx->VARIABLENAME()->getText();
        bool error = registerDeclaration( loc, varName );
        if ( error )
        {
            return;
        }

        var_states[varName] = variable_state::declared;
        // Allocate memref<f64> for the variable
        auto memrefType = mlir::MemRefType::get( {}, builder.getF64Type() );
        auto allocaOp =
            builder.create<mlir::memref::AllocaOp>( loc, memrefType );
        var_storage[varName] = allocaOp.getResult();
        builder.create<toy::DeclareOp>( loc, builder.getStringAttr( varName ) );
    }

    void MLIRListener::enterBoolDeclare( ToyParser::BoolDeclareContext *ctx )
    {
        lastOp = lastOperator::declareOp;
        auto loc = getLocation( ctx );
        auto varName = ctx->VARIABLENAME()->getText();
        bool error = registerDeclaration( loc, varName );
        if ( error )
        {
            return;
        }

        // Allocate memref<i1> for the variable
        auto memrefType = mlir::MemRefType::get( {}, builder.getI1Type() );
        auto allocaOp =
            builder.create<mlir::memref::AllocaOp>( loc, memrefType );
        var_storage[varName] = allocaOp.getResult();

        builder.create<toy::DeclareOp>( loc, builder.getStringAttr( varName ) );
    }

    void MLIRListener::enterIntDeclare( ToyParser::IntDeclareContext *ctx )
    {
        lastOp = lastOperator::declareOp;
        auto loc = getLocation( ctx );
        auto varName = ctx->VARIABLENAME()->getText();
        bool error = registerDeclaration( loc, varName );
        if ( error )
        {
            return;
        }

        // Allocate memref<...> for the variable
        if ( ctx->INT8() )
        {
            auto memrefType = mlir::MemRefType::get( {}, builder.getI8Type() );
            auto allocaOp =
                builder.create<mlir::memref::AllocaOp>( loc, memrefType );
            var_storage[varName] = allocaOp.getResult();
        }
        else if ( ctx->INT16() )
        {
            auto memrefType = mlir::MemRefType::get( {}, builder.getI16Type() );
            auto allocaOp =
                builder.create<mlir::memref::AllocaOp>( loc, memrefType );
            var_storage[varName] = allocaOp.getResult();
        }
        else if ( ctx->INT32() )
        {
            auto memrefType = mlir::MemRefType::get( {}, builder.getI32Type() );
            auto allocaOp =
                builder.create<mlir::memref::AllocaOp>( loc, memrefType );
            var_storage[varName] = allocaOp.getResult();
        }
        else if ( ctx->INT64() )
        {
            auto memrefType = mlir::MemRefType::get( {}, builder.getI64Type() );
            auto allocaOp =
                builder.create<mlir::memref::AllocaOp>( loc, memrefType );
            var_storage[varName] = allocaOp.getResult();
        }
        else
        {
            llvm::errs() << "Internal error: Unsupported signed integer "
                            "declaration size.\n";
            throw semantic_exception();
        }

        builder.create<toy::DeclareOp>( loc, builder.getStringAttr( varName ) );
    }

    void MLIRListener::enterFloatDeclare( ToyParser::FloatDeclareContext *ctx )
    {
        lastOp = lastOperator::declareOp;
        auto loc = getLocation( ctx );
        auto varName = ctx->VARIABLENAME()->getText();
        bool error = registerDeclaration( loc, varName );
        if ( error )
        {
            return;
        }

        // Allocate memref<...> for the variable
        if ( ctx->FLOAT32() )
        {
            auto memrefType = mlir::MemRefType::get( {}, builder.getF32Type() );
            auto allocaOp =
                builder.create<mlir::memref::AllocaOp>( loc, memrefType );
            var_storage[varName] = allocaOp.getResult();
        }
        else if ( ctx->FLOAT64() )
        {
            auto memrefType = mlir::MemRefType::get( {}, builder.getF64Type() );
            auto allocaOp =
                builder.create<mlir::memref::AllocaOp>( loc, memrefType );
            var_storage[varName] = allocaOp.getResult();
        }
        else
        {
            llvm::errs() << "Internal error: Unsupported floating point "
                            "declaration size.\n";
            throw semantic_exception();
        }
        builder.create<toy::DeclareOp>( loc, builder.getStringAttr( varName ) );
    }

    void MLIRListener::enterPrint( ToyParser::PrintContext *ctx )
    {
        lastOp = lastOperator::printOp;
        auto loc = getLocation( ctx );
        auto varName = ctx->VARIABLENAME()->getText();
        auto varState = var_states[varName];
        if ( varState == variable_state::undeclared )
        {
            lastSemError = semantic_errors::variable_not_declared;
            llvm::errs() << std::format(
                "{}error: Variable {} not declared in PRINT\n",
                formatLocation( loc ), varName );
            return;
        }
        if ( varState != variable_state::assigned )
        {
            lastSemError = semantic_errors::variable_not_assigned;
            llvm::errs() << std::format(
                "{}error: Variable {} not assigned in PRINT\n",
                formatLocation( loc ), varName );
            return;
        }

        auto memref = var_storage[varName];
        builder.create<toy::PrintOp>( loc, memref );
    }

    void MLIRListener::enterExitStatement(
        ToyParser::ExitStatementContext *ctx )
    {
        lastOp = lastOperator::returnOp;
        auto loc = getLocation( ctx );

        auto sz = ctx->unaryExpression().size();

        if ( sz == 0 )
        {
            //  Create toy.return with no operands (empty ValueRange)
            builder.create<toy::ExitOp>( loc, mlir::ValueRange{} );
        }
        else
        {
            mlir::Value value;
            assert( sz == 1 );

            auto n = ctx->unaryExpression()[0];

            assert( !n->unaryOperator()->MINUSCHAR() );    // TODO.

            auto lit = n->literal();

            bool error = buildUnaryExpression(
                lit ? lit->BOOLEANLITERAL() : nullptr,
                lit ? lit->INTEGERLITERAL() : nullptr,
                lit ? lit->FLOATLITERAL() : nullptr,
                lit ? n->VARIABLENAME() : nullptr, loc, value, false );
            if ( error )
            {
                return;
            }

            builder.create<toy::ExitOp>( loc, mlir::ValueRange{ value } );
        }
    }

    void MLIRListener::enterAssignment( ToyParser::AssignmentContext *ctx )
    {
        lastOp = lastOperator::assignmentOp;
        assignmentTargetValid = true;
        auto loc = getLocation( ctx );
        currentVarName = ctx->VARIABLENAME()->getText();
        auto varState = var_states[currentVarName];
        if ( varState == variable_state::declared )
        {
            var_states[currentVarName] = variable_state::assigned;
        }
        else if ( varState == variable_state::undeclared )
        {
            lastSemError = semantic_errors::variable_not_declared;
            llvm::errs() << std::format(
                "{}error: Variable {} not declared in assignment\n",
                formatLocation( loc ), currentVarName );
            assignmentTargetValid = false;
        }
        currentAssignLoc = loc;
    }


    void MLIRListener::enterUnaryExpression(
        ToyParser::UnaryExpressionContext *ctx )
    {
        if ( !assignmentTargetValid )
        {
            return;
        }

        auto loc = getLocation( ctx );
        auto op = ctx->unaryOperator()->getText();

        mlir::Value lhsValue;

        auto lit = ctx->literal();

        bool error = buildUnaryExpression(
            lit ? lit->BOOLEANLITERAL() : nullptr,
            lit ? lit->INTEGERLITERAL() : nullptr,
            lit ? lit->FLOATLITERAL() : nullptr,
            lit ? ctx->VARIABLENAME() : nullptr, loc, lhsValue, true );
        if ( error )
        {
            return;
        }

        mlir::Value resultValue;
        if ( op == "" || op == "+" )
        {
            resultValue = lhsValue;
        }
        else
        {
            auto negOp = builder.create<toy::NegOp>( loc, builder.getF64Type(),
                                                     lhsValue );
            resultValue = negOp.getResult();
        }

        assert( !currentVarName.empty() );

        // Store result to memref<f64>
        auto memref = var_storage[currentVarName];
        builder.create<mlir::memref::StoreOp>( loc, resultValue, memref );
        builder.create<toy::AssignOp>( currentAssignLoc,
                                       builder.getStringAttr( currentVarName ),
                                       resultValue );
        var_states[currentVarName] = variable_state::assigned;
        currentVarName.clear();
    }

    void MLIRListener::enterBinaryExpression(
        ToyParser::BinaryExpressionContext *ctx )
    {
        if ( !assignmentTargetValid )
        {
            return;
        }
        auto loc = getLocation( ctx );
        auto sz = ctx->binaryElement().size();
        auto lhs = ctx->binaryElement()[0];
        auto rhs = ctx->binaryElement()[1];
        auto op = ctx->binaryOperator()->getText();
        bool error;

        assert( sz == 2 );

        mlir::Value lhsValue;
        auto llit = lhs->numericLiteral();
        error = buildUnaryExpression(
            nullptr, llit ? llit->INTEGERLITERAL() : nullptr,
            llit ? llit->FLOATLITERAL() : nullptr,
            llit ? lhs->VARIABLENAME() : nullptr, loc, lhsValue, true );
        if ( error )
        {
            return;
        }

        mlir::Value rhsValue;
        auto rlit = rhs->numericLiteral();
        error = buildUnaryExpression(
            nullptr, llit ? llit->INTEGERLITERAL() : nullptr,
            llit ? llit->FLOATLITERAL() : nullptr,
            llit ? lhs->VARIABLENAME() : nullptr, loc, lhsValue, true );
        if ( error )
        {
            return;
        }

        // Create the binary operator (supports +, -, *, /)
        mlir::Value resultValue;
        switch ( op[0] )
        {
            case '+':
            {
                auto b = builder.create<toy::AddOp>( loc, builder.getF64Type(),
                                                     lhsValue, rhsValue );
                resultValue = b.getResult();
                break;
            }
            case '-':
            {
                auto b = builder.create<toy::SubOp>( loc, builder.getF64Type(),
                                                     lhsValue, rhsValue );
                resultValue = b.getResult();
                break;
            }
            case '*':
            {
                auto b = builder.create<toy::MulOp>( loc, builder.getF64Type(),
                                                     lhsValue, rhsValue );
                resultValue = b.getResult();
                break;
            }
            case '/':
            {
                auto b = builder.create<toy::DivOp>( loc, builder.getF64Type(),
                                                     lhsValue, rhsValue );
                resultValue = b.getResult();
                break;
            }
            default:
            {
                llvm::errs()
                    << std::format( "error: Invalid binary operator {}\n", op );
                throw semantic_exception();
            }
        }

        assert( !currentVarName.empty() );

        // Store result to memref<f64>
        auto memref = var_storage[currentVarName];
        builder.create<mlir::memref::StoreOp>( loc, resultValue, memref );
        builder.create<toy::AssignOp>( currentAssignLoc,
                                       builder.getStringAttr( currentVarName ),
                                       resultValue );
        var_states[currentVarName] = variable_state::assigned;
        currentVarName.clear();
    }
}    // namespace toy

// vim: et ts=4 sw=4
