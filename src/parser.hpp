/**
 * @file    parser.hpp
 * @author  Peeter Joot <peeterjoot@pm.me>
 * @brief   Antlr4 based Listener, and MLIR builder for the toy (calculator) compiler.
 *
 */
#pragma once

#include <antlr4-runtime.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>

#include <format>
#include <map>
#include <string>
#include <unordered_map>

#include "ToyBaseListener.h"
#include "ToyDialect.hpp"
#include "ToyExceptions.hpp"

namespace toy
{
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

    struct PerFunctionState
    {
        std::unordered_map<std::string, variable_state> varStates;
        mlir::Operation *funcOp{};
        mlir::Location lastLoc;
        bool terminatorWasExplcit{};

        PerFunctionState( mlir::Location loc ) : lastLoc( loc )
        {
        }
    };

    class MLIRListener : public ToyBaseListener, public antlr4::BaseErrorListener
    {
       public:
        MLIRListener( const std::string &_filename );

        // Override syntaxError to handle parsing errors
        void syntaxError( antlr4::Recognizer *recognizer, antlr4::Token *offendingSymbol, size_t line,
                          size_t charPositionInLine, const std::string &msg, std::exception_ptr e ) override;

        void enterStartRule( ToyParser::StartRuleContext *ctx ) override;

        void exitStartRule( ToyParser::StartRuleContext *ctx ) override;

        void exitIfStatement( ToyParser::IfStatementContext *ctx ) override;

        void exitElseStatement( ToyParser::ElseStatementContext *ctx ) override;

        void mainFirstTime( mlir::Location loc );

        void enterIfelifelse( ToyParser::IfelifelseContext *ctx ) override;

        void enterFunction( ToyParser::FunctionContext *ctx ) override;

        void enterCall( ToyParser::CallContext *ctx ) override;

        void exitFunction( ToyParser::FunctionContext *ctx ) override;

        void enterReturnStatement( ToyParser::ReturnStatementContext *ctx ) override;

        void enterDeclare( ToyParser::DeclareContext *ctx ) override;

        void enterBoolDeclare( ToyParser::BoolDeclareContext *ctx ) override;

        void enterIntDeclare( ToyParser::IntDeclareContext *ctx ) override;

        void enterFloatDeclare( ToyParser::FloatDeclareContext *ctx ) override;

        void enterStringDeclare( ToyParser::StringDeclareContext *ctx ) override;

        void enterPrint( ToyParser::PrintContext *ctx ) override;

        void enterGet( ToyParser::GetContext *ctx ) override;

        void enterFor( ToyParser::ForContext *ctx ) override;

        void exitFor( ToyParser::ForContext *ctx ) override;

        void enterAssignment( ToyParser::AssignmentContext *ctx ) override;

        void exitAssignment( ToyParser::AssignmentContext *ctx ) override;

        void enterExitStatement( ToyParser::ExitStatementContext *ctx ) override;

        void enterRhs( ToyParser::RhsContext *ctx ) override;

        inline mlir::ModuleOp &getModule();

       private:
        std::string filename;
        DialectCtx dialect;
        mlir::OpBuilder builder;
        mlir::Location currentAssignLoc;
        std::string currentFuncName;
        mlir::FileLineColLoc lastLocation;
        std::vector<mlir::OpBuilder::InsertPoint> insertionPointStack;    ///< scf.if block stack
        mlir::ModuleOp mod;
        std::string currentVarName;
        mlir::Value currentIndexExpr;
        std::unordered_map<std::string, std::unique_ptr<PerFunctionState>> pr_funcState;
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

        bool callIsHandled{};
        bool mainScopeGenerated{};

        inline toy::DeclareOp lookupDeclareForVar( const std::string &varName );

        inline mlir::Location getLocation( antlr4::ParserRuleContext *ctx, bool useStopLocation = false );

        void createScope( mlir::Location loc, mlir::func::FuncOp func, const std::string &funcName,
                          const std::vector<std::string> &paramNames );

        inline std::string formatLocation( mlir::Location loc );

        void buildUnaryExpression( tNode *booleanNode, tNode *integerNode, tNode *floatNode,
                                   ToyParser::ScalarOrArrayElementContext *scalarOrArrayElement, tNode *stringNode,
                                   mlir::Location loc, mlir::Value &value, std::string &s );

        mlir::Value handleCall( ToyParser::CallContext *ctx );

        // @param asz [in]
        //    Array size or zero for scalar.
        void registerDeclaration( mlir::Location loc, const std::string &varName, mlir::Type ty,
                                  ToyParser::ArrayBoundsExpressionContext *arrayBounds );

        inline PerFunctionState &funcState( const std::string &funcName );

        inline void setVarState( const std::string &funcName, const std::string &varName, variable_state st );

        inline variable_state getVarState( const std::string &varName );

        inline void setFuncOp( mlir::Operation *op );

        inline mlir::func::FuncOp getFuncOp( const std::string &funcName );

        inline void markExplicitTerminator();

        inline bool wasTerminatorExplicit();

        inline void setLastLoc( mlir::Location loc );

        inline mlir::Location getLastLoc();

        mlir::Type parseScalarType( const std::string &ty );

        mlir::Value castOpIfRequired( mlir::Location loc, mlir::Value value, mlir::Type desiredType );

        mlir::Value parsePredicate( mlir::Location loc, ToyParser::BooleanValueContext *ctx );

        mlir::Value indexTypeCast( mlir::Location loc, mlir::Value val );

        toy::ScopeOp getEnclosingScopeOp( mlir::func::FuncOp funcOp ) const;

        template <class Literal>
        void processReturnLike( mlir::Location loc, Literal *lit,
                                ToyParser::ScalarOrArrayElementContext *scalarOrArrayElement, tNode *boolNode );
    };

    inline mlir::ModuleOp &MLIRListener::getModule()
    {
        if ( hasErrors )
        {
            throw exception_with_context( __FILE__, __LINE__, __func__,
                                          std::format( "Cannot emit MLIR due to syntax errors in {}", filename ) );
        }

        return mod;
    }
}    // namespace toy

// vim: et ts=4 sw=4
