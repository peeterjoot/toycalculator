#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Types.h>

#include <format>

#include "ToyParser.h"
#include "ToyExceptions.h"

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
        if ( ctx )
        {
            size_t line = ctx->getStart()->getLine();
            size_t col = ctx->getStart()->getCharPositionInLine();
            return mlir::FileLineColLoc::get( builder.getStringAttr( filename ),
                                              line, col + 1 );
        }
        return mlir::UnknownLoc::get( &dialect.context );
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

    MLIRListener::MLIRListener( const std::string &_filename )
        : dialect(),
          builder( &dialect.context ),
          mod( mlir::ModuleOp::create( builder.getUnknownLoc() ) ),
          filename( _filename ),
          currentAssignLoc( builder.getUnknownLoc() )
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
        auto loc = getLocation( ctx );
        auto varName = ctx->VARIABLENAME()->getText();
        auto varState = var_states[varName];
        if ( varState == variable_state::declared )
        {
            lastSemError = semantic_errors::variable_already_declared;
            llvm::errs() << std::format(
                "{}error: Variable {} already declared in DCL\n",
                formatLocation( loc ), varName );
            return;
        }
        if ( varState == variable_state::undeclared )
        {
            var_states[varName] = variable_state::declared;
            // Allocate memref<f64> for the variable
            auto memrefType = mlir::MemRefType::get( {}, builder.getF64Type() );
            auto allocaOp =
                builder.create<mlir::memref::AllocaOp>( loc, memrefType );
            var_storage[varName] = allocaOp.getResult();
            builder.create<toy::DeclareOp>( loc,
                                            builder.getStringAttr( varName ) );
        }
    }

    void MLIRListener::enterPrint( ToyParser::PrintContext *ctx )
    {
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
        builder.create<toy::PrintOp>( loc, builder.getStringAttr( varName ) );
    }

    void MLIRListener::enterAssignment( ToyParser::AssignmentContext *ctx )
    {
        assignmentTargetValid = true;
        auto loc = getLocation( ctx );
        currentVarName = ctx->VARIABLENAME()->getText();
        auto varState = var_states[currentVarName];
        if ( varState == variable_state::declared )
        {
            var_states[currentVarName] = variable_state::assigned;
        }
        else if ( varState != variable_state::declared )
        {
            lastSemError = semantic_errors::variable_not_declared;
            llvm::errs() << std::format(
                "{}error: Variable {} not declared in assignment\n",
                formatLocation( loc ), currentVarName );
            assignmentTargetValid = false;
        }
        currentAssignLoc = loc;
    }

    void MLIRListener::enterUnaryexpression(
        ToyParser::UnaryexpressionContext *ctx )
    {
        if ( !assignmentTargetValid )
        {
            return;
        }
        auto loc = getLocation( ctx );
        auto lhs = ctx->element();
        auto op = ctx->unaryoperator()->getText();

        mlir::Value lhsValue;
        if ( lhs->INTEGERLITERAL() )
        {
            int64_t val = std::stoi( lhs->INTEGERLITERAL()->getText() );
            lhsValue =
                builder.create<mlir::arith::ConstantIntOp>( loc, val, 64 );
        }
        else if ( lhs->VARIABLENAME() )
        {
            auto varName = lhs->VARIABLENAME()->getText();
            auto varState = var_states[varName];
            if ( varState == variable_state::undeclared )
            {
                lastSemError = semantic_errors::variable_not_declared;
                llvm::errs() << std::format(
                    "{}error: Variable {} not declared in unary expr\n",
                    formatLocation( loc ), varName );
                return;
            }
            if ( varState != variable_state::assigned )
            {
                lastSemError = semantic_errors::variable_not_assigned;
                llvm::errs() << std::format(
                    "{}error: Variable {} not assigned in unary expr\n",
                    formatLocation( loc ), varName );
                return;
            }
            // Load from memref<f64>
            auto memref = var_storage[varName];
            lhsValue = builder.create<mlir::memref::LoadOp>( loc, memref );
        }
        else
        {
            lastSemError = semantic_errors::unknown_error;
            llvm::errs() << std::format( "{}error: Invalid unary operand\n",
                                         formatLocation( loc ) );
            return;
        }

        // Create UnaryOp (supports + or -)
        if ( op == "" )
        {
            op = "+";
        }

        auto unaryOp = builder.create<toy::UnaryOp>(
            loc, builder.getF64Type(), builder.getStringAttr( op ), lhsValue );

        if ( !currentVarName.empty() )
        {
            // Store result to memref<f64>
            auto memref = var_storage[currentVarName];
            builder.create<mlir::memref::StoreOp>( loc, unaryOp.getResult(),
                                                   memref );
            builder.create<toy::AssignOp>(
                currentAssignLoc, builder.getStringAttr( currentVarName ),
                unaryOp.getResult() );
            var_states[currentVarName] = variable_state::assigned;
            currentVarName.clear();
        }
    }

    void MLIRListener::enterBinaryexpression(
        ToyParser::BinaryexpressionContext *ctx )
    {
        if ( !assignmentTargetValid )
        {
            return;
        }
        auto loc = getLocation( ctx );
        auto sz = ctx->element().size();
        auto lhs = ctx->element()[0];
        auto rhs = ctx->element()[1];
        auto op = ctx->binaryoperator()->getText();

        assert( sz == 2 );

        mlir::Value lhsValue;
        if ( lhs->INTEGERLITERAL() )
        {
            int64_t val = std::stoi( lhs->INTEGERLITERAL()->getText() );
            lhsValue =
                builder.create<mlir::arith::ConstantIntOp>( loc, val, 64 );
        }
        else if ( lhs->VARIABLENAME() )
        {
            auto varName = lhs->VARIABLENAME()->getText();
            auto varState = var_states[varName];
            if ( varState == variable_state::undeclared )
            {
                lastSemError = semantic_errors::variable_not_declared;
                llvm::errs() << std::format(
                    "{}error: Variable {} not declared in binary expr\n",
                    formatLocation( loc ), varName );
                return;
            }
            if ( varState != variable_state::assigned )
            {
                lastSemError = semantic_errors::variable_not_assigned;
                llvm::errs() << std::format(
                    "{}error: Variable {} not assigned in binary expr\n",
                    formatLocation( loc ), varName );
                return;
            }
            // Load from memref<f64>
            auto memref = var_storage[varName];
            lhsValue = builder.create<mlir::memref::LoadOp>( loc, memref );
        }
        else
        {
            lastSemError = semantic_errors::unknown_error;
            llvm::errs() << std::format(
                "{}error: Invalid binary lhs operand\n",
                formatLocation( loc ) );
            return;
        }

        mlir::Value rhsValue;
        if ( rhs->INTEGERLITERAL() )
        {
            int64_t val = std::stoi( rhs->INTEGERLITERAL()->getText() );
            rhsValue =
                builder.create<mlir::arith::ConstantIntOp>( loc, val, 64 );
        }
        else if ( rhs->VARIABLENAME() )
        {
            auto varName = rhs->VARIABLENAME()->getText();
            auto varState = var_states[varName];
            if ( varState == variable_state::undeclared )
            {
                lastSemError = semantic_errors::variable_not_declared;
                llvm::errs() << std::format(
                    "{}error: Variable {} not declared in binary expr\n",
                    formatLocation( loc ), varName );
                return;
            }
            if ( varState != variable_state::assigned )
            {
                lastSemError = semantic_errors::variable_not_assigned;
                llvm::errs() << std::format(
                    "{}error: Variable {} not assigned in binary expr\n",
                    formatLocation( loc ), varName );
                return;
            }
            // Load from memref<f64>
            auto memref = var_storage[varName];
            rhsValue = builder.create<mlir::memref::LoadOp>( loc, memref );
        }
        else
        {
            lastSemError = semantic_errors::unknown_error;
            llvm::errs() << std::format(
                "{}error: Invalid binary rhs operand\n",
                formatLocation( loc ) );
            return;
        }

        // Create BinaryOp (supports +, -, *, /)
        auto binaryOp = builder.create<toy::BinaryOp>(
            loc, builder.getF64Type(), builder.getStringAttr( op ), lhsValue,
            rhsValue );

        if ( !currentVarName.empty() )
        {
            // Store result to memref<f64>
            auto memref = var_storage[currentVarName];
            builder.create<mlir::memref::StoreOp>( loc, binaryOp.getResult(),
                                                   memref );
            builder.create<toy::AssignOp>(
                currentAssignLoc, builder.getStringAttr( currentVarName ),
                binaryOp.getResult() );
            var_states[currentVarName] = variable_state::assigned;
            currentVarName.clear();
        }
    }
}    // namespace toy

// vim: et ts=4 sw=4
