/// @file    Builder.cpp
/// @author  Peeter Joot <peeterjoot@pm.me>
/// @brief   Grammar agnostic subset of the MLIR builder for the silly language.
#include <llvm/Support/FormatVariadic.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

#include <format>
#include <fstream>

#include "Builder.hpp"
#include "DriverState.hpp"
#include "LocationStack.hpp"
#include "SillyDialect.hpp"
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

    void Builder::registerDeclaration( mlir::Location loc, const std::string &varName, mlir::Type ty,
                                       mlir::Location aLoc, const std::string &arrayBounds, bool haveInitializers,
                                       std::vector<mlir::Value> &initializers, LocationStack &ls )
    {
        int64_t arraySize{};
        size_t numElements{ 1 };
        if ( !arrayBounds.empty() )
        {
            try
            {
                arraySize = std::stoi( arrayBounds );
            }
            catch ( std::exception &ex )
            {
                // coverage: syntax-error/too-big-array.silly
                emitUserError( aLoc,
                               std::format( "Array bounds integer literal value {} is out of range or bad: {}",
                                            arrayBounds, ex.what() ),
                               currentFuncName );
                return;
            }
            numElements = arraySize;
        }

        ParserPerFunctionState &f = funcState( currentFuncName );

        mlir::Value v = f.searchForVariable( varName );
        if ( v )
        {
            // coverage: syntax-error/redeclare.silly
            emitUserError( loc, std::format( "Variable {} already declared", varName ), currentFuncName );
            return;
        }

        if ( haveInitializers )
        {
            mlir::Value fill{};

            ssize_t remaining = numElements - initializers.size();

            if ( remaining )
            {
                if ( ty == typ.i1 )
                {
                    fill = parseBoolean( loc, "FALSE", ls );
                }
                else if ( mlir::IntegerType ity = mlir::dyn_cast<mlir::IntegerType>( ty ) )
                {
                    int width = ity.getWidth();
                    fill = parseInteger( loc, width, "0", ls );
                }
                else if ( mlir::FloatType fty = mlir::dyn_cast<mlir::FloatType>( ty ) )
                {
                    fill = parseFloat( loc, fty, "0", ls );
                }
                else
                {
                    emitInternalError( loc, __FILE__, __LINE__, __func__, "unknown scalar type.", currentFuncName );
                    return;
                }
            }

            if ( initializers.size() > numElements )
            {
                // coverage: syntax-error/array-too-many-init.silly, syntax-error/init-list-mismatch.silly,
                // syntax-error/init-list2.silly
                emitUserError(
                    loc,
                    std::format( "For variable '{}', more initializers ({}) specified than number of elements ({}).",
                                 varName, initializers.size(), numElements ),
                    currentFuncName );
                return;
            }

            for ( ssize_t i = 0; i < remaining; i++ )
            {
                assert( fill );
                initializers.push_back( fill );
            }
        }

        mlir::DenseI64ArrayAttr shapeAttr;
        if ( arraySize )
        {
            shapeAttr = builder.getDenseI64ArrayAttr( { arraySize } );
        }
        else
        {
            shapeAttr = builder.getDenseI64ArrayAttr( {} );
        }

        silly::varType varType = builder.getType<silly::varType>( ty, shapeAttr );

        silly::DeclareOp dcl = silly::DeclareOp::create( builder, loc, varType, initializers );
        f.recordVariableValue( varName, dcl.getResult() );

        f.setLastDeclared( dcl.getOperation() );

        mlir::Value debugScopeOp = f.currentDebugScope();

        silly::DebugNameOp::create( builder, loc, dcl.getResult(), varName, debugScopeOp );
    }

    silly::DeclareOp Builder::lookupDeclareForVar( mlir::Location loc, const std::string &varName )
    {
        silly::DeclareOp declareOp{};
        ParserPerFunctionState &f = funcState( currentFuncName );

        mlir::Value var = f.searchForVariable( varName );
        if ( !var )
        {
            // coverage: syntax-error/induction-in-step.silly
            emitUserError( loc, std::format( "Undeclared variable {}", varName ), currentFuncName );
            return declareOp;
        }

        declareOp = var.getDefiningOp<silly::DeclareOp>();
        assert( declareOp );    // not sure I could trigger NULL declareOp with user code.

        return declareOp;
    }

    mlir::Value Builder::variableToValue( mlir::Location loc, const std::string &varName, mlir::Value iValue,
                                          mlir::Location iLoc, LocationStack &ls )
    {
        mlir::Value value;
        ParserPerFunctionState &f = funcState( currentFuncName );
        mlir::Value iVar = f.searchForInduction( varName );
        mlir::Value pVar = f.searchForParameter( varName );
        if ( iVar )
        {
            value = iVar;
        }
        else if ( pVar )
        {
            value = pVar;
        }
        else
        {
            silly::DeclareOp declareOp = lookupDeclareForVar( loc, varName );
            if ( !declareOp )
            {
                emitInternalError( loc, __FILE__, __LINE__, __func__,
                                   std::format( "DeclareOp lookup for variable {} failed", varName ), currentFuncName );
                return value;
            }

            mlir::Value var = declareOp.getResult();
            silly::varType varTy = mlir::cast<silly::varType>( declareOp.getVar().getType() );
            mlir::Type elemType = varTy.getElementType();
            mlir::Value i{};

            if ( iValue )
            {
                i = indexTypeCast( iLoc, iValue, ls );

                ls.push_back( loc );
                value = silly::LoadOp::create( builder, loc, mlir::TypeRange{ elemType }, var, i );
            }
            else
            {
                mlir::DenseI64ArrayAttr shapeAttr = varTy.getShape();
                llvm::ArrayRef<int64_t> shape = shapeAttr.asArrayRef();

                if ( !shape.empty() )
                {
                    if ( mlir::IntegerType ity = mlir::cast<mlir::IntegerType>( elemType ) )
                    {
                        unsigned w = ity.getWidth();
                        if ( w == 8 )
                        {
                            elemType = typ.ptr;    // HACK.  Assumes that the only use of INT8[] is for STRING.
                        }
                    }
                }

                ls.push_back( loc );
                value = silly::LoadOp::create( builder, loc, mlir::TypeRange{ elemType }, var, i );
            }
        }

        return value;
    }

    mlir::Value Builder::indexTypeCast( mlir::Location loc, mlir::Value val, LocationStack &ls )
    {
        mlir::IndexType indexTy = builder.getIndexType();
        mlir::Type valTy = val.getType();

        if ( valTy == indexTy )
        {
            return val;
        }

        if ( !valTy.isSignlessInteger( 64 ) && valTy.isInteger() )
        {
            val = castOpIfRequired( loc, val, typ.i64, ls );
            valTy = typ.i64;
        }

        // Only support i64, or castable to i64, for now
        if ( !valTy.isSignlessInteger( 64 ) )
        {
            // If it's a non-i64 IntegerType, we could cast up to i64, and then cast that to index.
            emitInternalError(
                loc, __FILE__, __LINE__, __func__,
                std::format( "NYI: indexTypeCast from type {} is not supported.", mlirTypeToString( valTy ) ),
                currentFuncName );
            return mlir::Value{};
        }

        ls.push_back( loc );
        return mlir::arith::IndexCastOp::create( builder, loc, indexTy, val );
    }

    mlir::Value Builder::castOpIfRequired( mlir::Location loc, mlir::Value value, mlir::Type desiredType,
                                           LocationStack &ls )
    {
        mlir::Value newValue{};

        if ( value.getType() != desiredType )
        {
            mlir::Type valType = value.getType();

            if ( valType.isF64() )
            {
                if ( mlir::isa<mlir::IntegerType>( desiredType ) )
                {
                    newValue = mlir::arith::FPToSIOp::create( builder, loc, desiredType, value );
                }
                else if ( desiredType.isF32() )
                {
                    newValue = mlir::LLVM::FPExtOp::create( builder, loc, desiredType, value );
                }
            }
            else if ( valType.isF32() )
            {
                if ( mlir::isa<mlir::IntegerType>( desiredType ) )
                {
                    newValue = mlir::arith::FPToSIOp::create( builder, loc, desiredType, value );
                }
                else if ( desiredType.isF64() )
                {
                    newValue = mlir::LLVM::FPExtOp::create( builder, loc, desiredType, value );
                }
            }
            else if ( mlir::IntegerType viType = mlir::cast<mlir::IntegerType>( valType ) )
            {
                unsigned vwidth = viType.getWidth();
                if ( mlir::isa<mlir::FloatType>( desiredType ) )
                {
                    if ( vwidth == 1 )
                    {
                        newValue = mlir::arith::UIToFPOp::create( builder, loc, desiredType, value );
                    }
                    else
                    {
                        newValue = mlir::arith::SIToFPOp::create( builder, loc, desiredType, value );
                    }
                }
                else if ( mlir::IntegerType miType = mlir::cast<mlir::IntegerType>( desiredType ) )
                {
                    unsigned mwidth = miType.getWidth();
                    if ( ( vwidth == 1 ) && ( mwidth != 1 ) )
                    {
                        // widen bool to integer using unsigned extension:
                        newValue = mlir::arith::ExtUIOp::create( builder, loc, desiredType, value );
                    }
                    else if ( vwidth > mwidth )
                    {
                        newValue = mlir::arith::TruncIOp::create( builder, loc, desiredType, value );
                    }
                    else if ( vwidth < mwidth )
                    {
                        newValue = mlir::arith::ExtSIOp::create( builder, loc, desiredType, value );
                    }
                }
            }
        }

        if ( newValue )
        {
            ls.push_back( loc );
            return newValue;
        }

        return value;
    }

    void Builder::processAssignment( mlir::Location loc, mlir::Value resultValue, const std::string &currentVarName,
                                     mlir::Value currentIndexExpr, LocationStack &ls )
    {
        if ( !resultValue )
        {
            emitInternalError( loc, __FILE__, __LINE__, __func__, "no resultValue for expression", currentFuncName );
            return;
        }

        silly::DeclareOp declareOp = lookupDeclareForVar( loc, currentVarName );
        mlir::Value var = declareOp.getResult();

        assert( resultValue );

        mlir::BlockArgument ba = mlir::dyn_cast<mlir::BlockArgument>( resultValue );
        mlir::Operation *op = resultValue.getDefiningOp();
        mlir::Value i{};

        // mlir::Location fusedLoc = ls.fuseLocations( );

        // Don't check if it's a StringLiteralOp if it's an induction variable, since op will be nullptr
        if ( !ba && isa<silly::StringLiteralOp>( op ) )
        {
            silly::AssignOp::create( builder, loc, var, i, resultValue );
        }
        else
        {
            if ( currentIndexExpr )
            {
                mlir::Location loc = currentIndexExpr.getLoc();

                mlir::Value i = indexTypeCast( loc, currentIndexExpr, ls );

                silly::AssignOp assign = silly::AssignOp::create( builder, loc, var, i, resultValue );

                LLVM_DEBUG( {
                    mlir::OpPrintingFlags flags;
                    flags.enableDebugInfo( true );

                    assign->print( llvm::outs(), flags );
                    llvm::outs() << "\n";
                } );
            }
            else
            {
                silly::AssignOp::create( builder, loc, var, i, resultValue );
            }
        }
    }

    bool Builder::isVariableDeclared( const std::string &varName )
    {
        // Get the single scope
        ParserPerFunctionState &f = funcState( currentFuncName );

        mlir::Value v = f.searchForVariable( varName );
        return ( v != nullptr ) ? true : false;
    }

    mlir::Type Builder::findReturnType()
    {
        mlir::Type returnType{};
        if ( currentFuncName == ENTRY_SYMBOL_NAME )
        {
            returnType = typ.i32;
        }
        else
        {
            ParserPerFunctionState &f = funcState( currentFuncName );
            mlir::func::FuncOp funcOp = f.getFuncOp();
            llvm::ArrayRef<mlir::Type> returnTypeArray = funcOp.getFunctionType().getResults();

            if ( !returnTypeArray.empty() )
            {
                returnType = returnTypeArray[0];
            }
        }

        return returnType;
    }

    void Builder::processReturnLike( mlir::Location loc, mlir::Value returnValue, LocationStack &ls )
    {
        mlir::Value value{};
        ls.push_back( loc );

        if ( returnValue )
        {
            value = returnValue;
        }
        else if ( currentFuncName == ENTRY_SYMBOL_NAME )
        {
            value = mlir::arith::ConstantIntOp::create( builder, loc, 0, 32 );
        }

        // mlir::Location fused = ls.fuseLocations( );

        if ( value )
        {
            // Create ReturnOp with user specified value:
            mlir::func::ReturnOp::create( builder, loc, mlir::ValueRange{ value } );
        }
        else
        {
            mlir::func::ReturnOp::create( builder, loc, mlir::ValueRange{} );
        }
    }

    mlir::Value Builder::createBinaryArith( mlir::Location loc, silly::ArithBinOpKind what, mlir::Type ty,
                                            mlir::Value lhs, mlir::Value rhs, LocationStack &ls )
    {
        ls.push_back( loc );

        return silly::ArithBinOp::create( builder, loc, ty, silly::ArithBinOpKindAttr::get( this->ctx, what ), lhs,
                                          rhs )
            .getResult();
    }

    mlir::Value Builder::createBinaryCmp( mlir::Location loc, silly::CmpBinOpKind what, mlir::Value lhs,
                                          mlir::Value rhs, LocationStack &ls )
    {
        ls.push_back( loc );

        return silly::CmpBinOp::create( builder, loc, typ.i1, silly::CmpBinOpKindAttr::get( this->ctx, what ), lhs,
                                        rhs )
            .getResult();
    }

    mlir::Value Builder::makeUnaryExpression( mlir::Location loc, mlir::Value value, UnaryOp op, LocationStack &ls )
    {
        mlir::Value v;

        switch ( op )
        {
            case UnaryOp::Negate:
            {
                // Negation
                value = silly::NegOp::create( builder, loc, value.getType(), value ).getResult();
                break;
            }
            case UnaryOp::Not:
            {
                if ( !value.getType().isInteger() )
                {
                    // coverage: syntax-error/not-float.silly
                    emitUserError( loc, "NOT on non-integer type", currentFuncName );
                    return v;
                }

                // NOT x: (x == 0)
                mlir::Value zero =
                    mlir::arith::ConstantIntOp::create( builder, loc, 0, value.getType().getIntOrFloatBitWidth() );
                v = createBinaryCmp( loc, silly::CmpBinOpKind::Equal, value, zero, ls );
                break;
            }
            case UnaryOp::Plus:
            {
                v = value;
                break;
            }
            default:
            {
                emitInternalError( loc, __FILE__, __LINE__, __func__,
                                   std::format( "unknown unary operator: {}", (int)op ), currentFuncName );
            }
        }

        return v;
    }
}    // namespace silly

// vim: et ts=4 sw=4
