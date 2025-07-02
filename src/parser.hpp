/**
 * @file    parser.hpp
 * @author  Peeter Joot <peeterjoot@pm.me>
 * @brief   Antlr4 based Listener, implementing MLIR builder for the toy
 * (calculator) compiler.
 *
 */
#if !defined __ToyParser_hpp_is_included
#define __ToyParser_hpp_is_included

#pragma once

#include <antlr4-runtime.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>

#include <format>
#include <map>
#include <string>
#include <unordered_map>

#include "ToyBaseListener.h"
#include "ToyDialect.hpp"
#include "ToyExceptions.hpp"

namespace toy
{
    enum class theTypes : int
    {
        unknown,
        boolean,
        integer8,
        integer16,
        integer32,
        integer64,
        float32,
        float64,
        string
    };

    inline bool isBoolean( theTypes ty )
    {
        return ty == theTypes::boolean;
    }

    inline bool isInteger( theTypes ty )
    {
        return !isBoolean( ty ) && ( (int)ty < (int)theTypes::float32 );
    }

    inline bool isFloat( theTypes ty )
    {
        return (int)ty >= (int)theTypes::float32;
    }

    enum class semantic_errors : int
    {
        not_an_error,
        variable_already_declared,
        variable_not_declared,
        variable_not_assigned,
        unknown_error
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

    using tNode = antlr4::tree::TerminalNode;

    class MLIRListener : public ToyBaseListener, public antlr4::BaseErrorListener
    {
       private:
        std::string filename;
        DialectCtx dialect;
        mlir::OpBuilder builder;
        mlir::Location currentAssignLoc;
        std::string currentFuncName;
        mlir::FileLineColLoc lastLocation;
        mlir::ModuleOp mod;
        std::string currentVarName;
        semantic_errors lastSemError{ semantic_errors::not_an_error };
        std::unordered_map<std::string, variable_state> pr_varStates;
        std::map<std::string, mlir::Operation*> funcByName;
        bool assignmentTargetValid{};
        bool hasErrors{};

        mlir::IntegerType tyI1;
        mlir::IntegerType tyI8;
        mlir::IntegerType tyI16;
        mlir::IntegerType tyI32;
        mlir::IntegerType tyI64;
        mlir::FloatType tyF32;
        mlir::FloatType tyF64;
        mlir::LLVM::LLVMPointerType tyPtr;
        mlir::LLVM::LLVMVoidType tyVoid;

        mlir::OpBuilder::InsertPoint mainIP;

        inline toy::DeclareOp lookupDeclareForVar( const std::string & varName );

        inline mlir::Location getLocation( antlr4::ParserRuleContext *ctx );

        void createScope( mlir::Location loc, mlir::func::FuncOp func, const std::string & funcName, const std::vector<std::string> & paramNames );

        inline std::string formatLocation( mlir::Location loc );

        inline std::string buildUnaryExpression( tNode *booleanNode, tNode *integerNode, tNode *floatNode,
                                                 tNode *variableNode, tNode *stringNode, mlir::Location loc,
                                                 mlir::Value &value );

        // @param asz [in]
        //    Array size or zero for scalar.
        inline bool registerDeclaration( mlir::Location loc, const std::string &varName, mlir::Type ty,
                                         ToyParser::ArrayBoundsExpressionContext *arrayBounds );

        void setVarState( const std::string & funcName, const std::string & varName, variable_state st )
        {
            auto k = funcName + "::" + varName;
            pr_varStates[ k ] = st;
        }

        variable_state getVarState( const std::string & funcName, const std::string & varName )
        {
            auto k = funcName + "::" + varName;
            return pr_varStates[ k ];
        }

        mlir::Type parseScalarType( const std::string &ty );

       public:
        MLIRListener( const std::string &_filename );

        // Override syntaxError to handle parsing errors

        void syntaxError( antlr4::Recognizer *recognizer, antlr4::Token *offendingSymbol, size_t line,
                          size_t charPositionInLine, const std::string &msg, std::exception_ptr e) override
        {
            hasErrors = true;
            std::string tokenText = offendingSymbol ? offendingSymbol->getText() : "<none>";
            throw exception_with_context( __FILE__, __LINE__, __func__,
                                          std::format( "Syntax error in {}:{}:{}: {} (token: {} )", filename, line,
                                                       charPositionInLine, msg, tokenText ) );
        }

        mlir::ModuleOp &getModule()
        {
            if ( hasErrors )
            {
                throw exception_with_context( __FILE__, __LINE__, __func__,
                                              std::format( "Cannot emit MLIR due to syntax errors in {}", filename ) );
            }

            return mod;
        }

        template <class Literal>
        void processReturnLike( mlir::Location loc, Literal *lit, tNode *var, tNode *boolNode );

        void enterStartRule( ToyParser::StartRuleContext *ctx ) override;

        void enterIfelifelse( ToyParser::IfelifelseContext *ctx ) override;

        void enterFunction( ToyParser::FunctionContext *ctx ) override;

        void enterCall( ToyParser::CallContext *ctx) override;

        void exitFunction( ToyParser::FunctionContext *ctx ) override;

        void enterReturnStatement( ToyParser::ReturnStatementContext *ctx ) override;

        void enterDeclare( ToyParser::DeclareContext *ctx ) override;

        void enterBoolDeclare( ToyParser::BoolDeclareContext *ctx ) override;

        void enterIntDeclare( ToyParser::IntDeclareContext *ctx ) override;

        void enterFloatDeclare( ToyParser::FloatDeclareContext *ctx ) override;

        void enterStringDeclare( ToyParser::StringDeclareContext *ctx ) override;

        void enterPrint( ToyParser::PrintContext *ctx ) override;

        void enterAssignment( ToyParser::AssignmentContext *ctx ) override;

        void enterExitStatement( ToyParser::ExitStatementContext *ctx ) override;

        void enterAssignmentExpression( ToyParser::AssignmentExpressionContext *ctx ) override;
    };
}    // namespace toy

#endif

// vim: et ts=4 sw=4
