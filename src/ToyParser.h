/**
 * @file    ToyParser.h
 * @author  Peeter Joot <peeterjoot@pm.me>
 * @brief   Antlr4 based Listener, implementing MLIR builder for the toy
 * (calculator) compiler.
 *
 */
#if !defined __ToyParser_h_is_included
#define __ToyParser_h_is_included

#pragma once

#include <antlr4-runtime.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>

#include <map>
#include <string>
#include <unordered_map>

#include "ToyBaseListener.h"
#include "ToyDialect.h"

namespace toy
{

    enum class semantic_errors : int
    {
        not_an_error,
        variable_already_declared,
        variable_not_declared,
        variable_not_assigned,
        unknown_error
    };

    enum class lastOperator : int
    {
        notAnOp,
        declareOp,
        printOp,
        assignmentOp,
        returnOp
    };

    enum class variable_state : int
    {
        undeclared,
        declared,
        assigned
    };

    class DialectCtx
    {
       public:
        mlir::MLIRContext context;

        DialectCtx();
    };

    class MLIRListener : public ToyBaseListener
    {
       private:
        DialectCtx dialect;
        mlir::OpBuilder builder;
        mlir::ModuleOp mod;
        std::string filename;
        toy::ProgramOp programOp;
        std::string currentVarName;
        mlir::Location currentAssignLoc;
        semantic_errors lastSemError{ semantic_errors::not_an_error };
        std::unordered_map<std::string, variable_state> var_states;
        std::map<std::string, mlir::Value>
            var_storage;    // Maps variable names to memref<f64>
        bool assignmentTargetValid;
        lastOperator lastOp{lastOperator::notAnOp};

        inline mlir::Location getLocation( antlr4::ParserRuleContext *ctx );

        inline std::string formatLocation( mlir::Location loc );

       public:
        MLIRListener( const std::string &_filename );

        mlir::ModuleOp &getModule()
        {
            return mod;
        }

        void enterStartRule( ToyParser::StartRuleContext *ctx ) override;

        void exitStartRule( ToyParser::StartRuleContext *ctx ) override;

        void enterDeclare( ToyParser::DeclareContext *ctx ) override;

        void enterPrint( ToyParser::PrintContext *ctx ) override;

        void enterAssignment( ToyParser::AssignmentContext *ctx ) override;

        void enterReturn( ToyParser::ReturnContext *ctx ) override;

        void enterUnaryexpression(
            ToyParser::UnaryexpressionContext *ctx ) override;

        void enterBinaryexpression(
            ToyParser::BinaryexpressionContext *ctx ) override;
    };
}    // namespace toy

#endif

// vim: et ts=4 sw=4
