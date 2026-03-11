/// @file    Builder.cpp
/// @author  Peeter Joot <peeterjoot@pm.me>
/// @brief   Grammar agnostic subset of the MLIR builder for the silly language.
#include <llvm/Support/FormatVariadic.h>
#include <mlir/Dialect/Arith/IR/Arith.h>

#include <format>
#include <fstream>

#include "Builder.hpp"
#include "DriverState.hpp"
#include "SillyDialect.hpp"
#include "LocationStack.hpp"
#include "SourceManager.hpp"
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

    void Builder::createScope( mlir::Location startLoc, mlir::func::FuncOp funcOp, const std::string &funcName,
                               const std::vector<std::string> &paramNames )
    {
        LLVM_DEBUG( {
            llvm::errs() << llvm::formatv( "createScope: {0}: startLoc: {1}\n", funcName, formatLocation( startLoc ) );
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

    void Builder::emitInternalError( mlir::Location loc, const char *compilerfile, unsigned compilerline,
                                     const char *compilerfunc, const std::string &message,
                                     const std::string &programFuncName )
    {
        emitError( loc, std::format( "{}:{}:{}: {}", compilerfile, compilerline, compilerfunc, message ),
                   programFuncName, true );
    }

    void Builder::emitError( mlir::Location loc, const std::string &message, const std::string &funcName,
                             bool internal )
    {
        bool inColor = isatty( fileno( stderr ) ) && !driverState.noColorErrors;
        const char *RED = inColor ? "\033[1;31m" : "";
        const char *CYAN = inColor ? "\033[0;36m" : "";
        const char *RESET = inColor ? "\033[0m" : "";

        if ( internal && errorCount )
        {
            errorCount++;
            return;
        }

        static std::string lastFunc{};
        auto fileLoc = mlir::dyn_cast<mlir::FileLineColLoc>( loc );
        if ( !fileLoc )
        {
            llvm::errs() << llvm::formatv( "{0}{1}error: {2}{3}\n", RED, internal ? "internal " : "", RESET, message );
        }

        std::string sourcename = fileLoc.getFilename().str();
        unsigned line = fileLoc.getLine();
        unsigned col = fileLoc.getColumn();

        if ( ( funcName != "" ) && ( funcName != ENTRY_SYMBOL_NAME ) && ( funcName != lastFunc ) )
        {
            llvm::errs() << llvm::formatv( "{0}: In function ‘{1}’:\n", sourcename, funcName );
        }
        lastFunc = funcName;

        // Print: sourcename:line:col: error: message
        llvm::errs() << llvm::formatv( "{0}{1}:{2}:{3}: {4}{5}error: {6}{7}\n", CYAN, sourcename, line, col, RED,
                                       internal ? "internal " : "", RESET, message );

        // Try to read and display the source line
        if ( !sourcename.empty() )
        {
            std::string path = sourcename;

            if ( std::ifstream file{ path } )
            {
                std::string currentLine;
                unsigned currentLineNum = 0;

                while ( std::getline( file, currentLine ) )
                {
                    currentLineNum++;
                    if ( currentLineNum == line )
                    {
                        /* Example output:
                            5 | FOR(INT32 i:(0,2)){PRINT i;}
                              |     ^
                         */
                        // `{:>{}}` - the `^` character, right-aligned to `col` width
                        llvm::errs() << std::format(
                            "{0:5} | {1}\n"
                            "      | {2:>{3}}\n",
                            line, currentLine, "^", col );
                        break;
                    }
                }
            }
        }

        errorCount++;
    }

    mlir::Value Builder::parseBoolean( mlir::Location loc, const std::string &s, LocationStack &ls )
    {
        int val;
        if ( s == "TRUE" )
        {
            val = 1;
        }
        else if ( s == "FALSE" )
        {
            val = 0;
        }
        else
        {
            emitInternalError( loc, __FILE__, __LINE__, __func__,
                               std::format( "boolean value neither TRUE nor FALSE: {}", s ), currentFuncName );
            return mlir::Value{};
        }

        ls.push_back( loc );
        return mlir::arith::ConstantIntOp::create( builder, loc, val, 1 );
    }

    mlir::Value Builder::parseInteger( mlir::Location loc, int width, const std::string &s, LocationStack &ls )
    {
        int64_t val{};

        try
        {
            val = std::stoll( s );
        }
        catch ( std::exception &ex )
        {
            // coverage: syntax-error/toolong-int-literal.silly
            emitUserError( loc, std::format( "Integer literal value {} is out of range or bad: {}", s, ex.what() ),
                           currentFuncName );
            return nullptr;
        }

        ls.push_back( loc );
        return mlir::arith::ConstantIntOp::create( builder, loc, val, width );
    }

    mlir::Value Builder::parseFloat( mlir::Location loc, mlir::FloatType ty, const std::string &s, LocationStack &ls )
    {
        ls.push_back( loc );

        try
        {
            if ( ty == typ.f32 )
            {
                float val = std::stof( s );

                llvm::APFloat apVal( val );

                return mlir::arith::ConstantFloatOp::create( builder, loc, typ.f32, apVal );
            }
            else
            {
                double val = std::stod( s );

                llvm::APFloat apVal( val );

                return mlir::arith::ConstantFloatOp::create( builder, loc, typ.f64, apVal );
            }
        }
        catch ( std::exception &ex )
        {
            // coverage: syntax-error/toolong-float-literal.silly
            emitUserError( loc,
                           std::format( "Floating point literal value {} is out of range or bad: {}", s, ex.what() ),
                           currentFuncName );
            return nullptr;
        }
    }

    silly::StringLiteralOp Builder::buildStringLiteral( mlir::Location loc, const std::string &input,
                                                        LocationStack &ls )
    {
        silly::StringLiteralOp stringLiteral{};

        if ( ( input.size() < 2 ) || ( input.front() != '"' ) || ( input.back() != '"' ) )
        {
            emitInternalError( loc, __FILE__, __LINE__, __func__,
                               std::format( "String '{}' was not double quotes enclosed as expected.", input ),
                               currentFuncName );
            return stringLiteral;
        }

        std::string s = input.substr( 1, input.size() - 2 );

        mlir::StringAttr strAttr = builder.getStringAttr( s );

        ls.push_back( loc );
        stringLiteral = silly::StringLiteralOp::create( builder, loc, typ.ptr, strAttr );

        return stringLiteral;
    }
}    // namespace silly

// vim: et ts=4 sw=4
