/**
 * @file    hack_builder.hpp
 * @author  Peeter Joot <peeterjoot@pm.me>
 * @brief   MLIR builder for builder/lowering experiments.
 */
#if !defined __hack_builder_is_included
#define __hack_builder_is_included

#pragma once

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>

#include <format>
#include <map>
#include <string>
#include <unordered_map>

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

    class MLIRListener
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

        inline mlir::Location getLocation( void * ctx = nullptr );

        void createScope( mlir::Location loc, mlir::func::FuncOp func, const std::string & funcName );

        inline std::string formatLocation( mlir::Location loc );

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

        mlir::ModuleOp &getModule()
        {
            return mod;
        }

        template <class Literal>
        void processReturnLike( mlir::Location loc );

        void enterStartRule(  ) ;

        void enterIfelifelse(  ) ;

        void enterFunction(  ) ;

        void enterCall( ) ;

        void exitFunction(  ) ;

        void enterReturnStatement(  ) ;

        void enterDeclare(  ) ;

        void enterBoolDeclare(  ) ;

        void enterIntDeclare(  ) ;

        void enterFloatDeclare(  ) ;

        void enterStringDeclare(  ) ;

        void enterPrint(  ) ;

        void enterAssignment(  ) ;

        void enterExitStatement(  ) ;

        void enterAssignmentExpression(  ) ;
    };
}    // namespace toy

#endif

// vim: et ts=4 sw=4
