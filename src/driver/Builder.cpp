/// @file    Builder.cpp
/// @author  Peeter Joot <peeterjoot@pm.me>
/// @brief   Grammar agnostic subset of the MLIR builder for the silly language.
#include <llvm/Support/FormatVariadic.h>
#include <mlir/Dialect/Arith/IR/Arith.h>

#include "Builder.hpp"
#include "DriverState.hpp"
#include "SourceManager.hpp"
#include "SillyDialect.hpp"
#include "helper.hpp"    // formatLocation

/// --debug- class for the builder
#define DEBUG_TYPE "silly-builder"

namespace silly
{
    Builder::Builder( silly::SourceManager &s, const std::string &filename )
        : sm{ s },
          driverState{ sm.getDriverState() },
          sourceFile{ filename },
          ctx{ sm.getContext() },
          builder( ctx ),
          rmod( mlir::ModuleOp::create( mlir::FileLineColLoc::get( builder.getStringAttr( filename ), 0, 0 ) ) ),
          mainIP{},
          currentFuncName{},
          functionStateMap{}
    {
        builder.setInsertionPointToStart( rmod->getBody() );
        typ.initialize( builder, ctx );
    }

    ParserPerFunctionState &Builder::funcState( const std::string &funcName )
    {
        if ( !functionStateMap.contains( funcName ) )
        {
            functionStateMap[funcName] = std::make_unique<ParserPerFunctionState>();
        }

        auto &p = functionStateMap[funcName];

        return *p;
    }

    void Builder::createScope( mlir::Location startLoc, mlir::func::FuncOp funcOp,
                               const std::string &funcName, const std::vector<std::string> &paramNames )
    {
        LLVM_DEBUG( {
            llvm::errs() << llvm::formatv( "createScope: {0}: startLoc: {1}\n", funcName,
                                           formatLocation( startLoc ) );
        } );
        mlir::Block &block = *funcOp.addEntryBlock();
        builder.setInsertionPointToStart( &block );

        ParserPerFunctionState &f = funcState( funcName );

        for ( size_t i = 0; i < funcOp.getNumArguments() && i < paramNames.size(); ++i )
        {
            LLVM_DEBUG( {
                llvm::errs() << llvm::formatv( "function {0}: parameter{1}:\n", funcName, i );
                funcOp.getArgument( i ).dump();
            } );

            mlir::Value param = funcOp.getArgument( i );
            silly::DebugNameOp::create( builder, startLoc, param, paramNames[i], f.currentDebugScope() );
            f.recordParameterValue( paramNames[i], param );
        }

        currentFuncName = funcName;
        f.setFuncOp( funcOp );

        if ( funcName == ENTRY_SYMBOL_NAME )
        {
            f.pushScopeOp( mlir::Value{} );
        }
    }

    void Builder::createMain( mlir::Location fLoc, mlir::Location sLoc )
    {
        mlir::FunctionType funcType = builder.getFunctionType( {}, typ.i32 );
        mlir::func::FuncOp funcOp = mlir::func::FuncOp::create( builder, fLoc, ENTRY_SYMBOL_NAME, funcType );
        std::vector<std::string> paramNames;
        createScope( sLoc, funcOp, ENTRY_SYMBOL_NAME, paramNames );
    }

    void Builder::createMainExit( mlir::Location loc )
    {
        assert( currentFuncName == ENTRY_SYMBOL_NAME );

        LLVM_DEBUG( {
            llvm::errs() << llvm::formatv( "exitStartRule: implicit exit generation with location: {0}:\n",
                                           formatLocation( loc ) );
        } );

        mlir::Value value = mlir::arith::ConstantIntOp::create( builder, loc, 0, 32 );
        mlir::func::ReturnOp::create( builder, loc, mlir::ValueRange{ value } );
    }
}    // namespace silly

// vim: et ts=4 sw=4
