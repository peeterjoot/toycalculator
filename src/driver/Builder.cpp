/// @file    Builder.cpp
/// @author  Peeter Joot <peeterjoot@pm.me>
/// @brief   Grammar agnostic subset of the MLIR builder for the silly language.
#include <llvm/Support/FormatVariadic.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/SCF/IR/SCF.h>

#include <format>
#include <fstream>

#include "Builder.hpp"
#include "DriverState.hpp"
#include "LocationStack.hpp"
#include "ModuleInsertionPointGuard.hpp"
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

    ParserPerFunctionState &Builder::lookupFunctionState( const std::string &funcName )
    {
        if ( !functionStateMap.contains( funcName ) )
        {
            functionStateMap[funcName] = std::make_unique<ParserPerFunctionState>();
        }

        auto &p = functionStateMap[funcName];

        return *p;
    }

    void Builder::createNewFunctionState( mlir::Location startLoc, mlir::func::FuncOp funcOp,
                                          const std::string &funcName, const std::vector<std::string> &paramNames )
    {
        LLVM_DEBUG( {
            llvm::errs() << llvm::formatv( "createNewFunctionState: {0}: startLoc: {1}\n", funcName,
                                           formatLocation( startLoc ) );
        } );
        mlir::Block &block = *funcOp.addEntryBlock();
        builder.setInsertionPointToStart( &block );

        ParserPerFunctionState &f = lookupFunctionState( funcName );

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
        createNewFunctionState( sLoc, funcOp, ENTRY_SYMBOL_NAME, paramNames );
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
            errorCount++;
            return;
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

    mlir::Value Builder::createBooleanFromString( mlir::Location loc, const std::string &s, LocationStack &ls )
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

    mlir::Value Builder::createIntegerFromString( mlir::Location loc, int width, const std::string &s,
                                                  LocationStack &ls )
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

    mlir::Value Builder::createFloatFromString( mlir::Location loc, mlir::FloatType ty, const std::string &s,
                                                LocationStack &ls )
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

    silly::StringLiteralOp Builder::createStringLiteral( mlir::Location loc, const std::string &input,
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

    void Builder::createDeclaration( mlir::Location loc, const std::string &varName, mlir::Type ty, mlir::Location aLoc,
                                     const std::string &arrayBounds, bool haveInitializers,
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

        ParserPerFunctionState &f = lookupFunctionState( currentFuncName );

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
                    fill = createBooleanFromString( loc, "FALSE", ls );
                }
                else if ( mlir::IntegerType ity = mlir::dyn_cast<mlir::IntegerType>( ty ) )
                {
                    int width = ity.getWidth();
                    fill = createIntegerFromString( loc, width, "0", ls );
                }
                else if ( mlir::FloatType fty = mlir::dyn_cast<mlir::FloatType>( ty ) )
                {
                    fill = createFloatFromString( loc, fty, "0", ls );
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
        ParserPerFunctionState &f = lookupFunctionState( currentFuncName );

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

    mlir::Value Builder::createVariableLoad( mlir::Location loc, const std::string &varName, mlir::Value iValue,
                                             mlir::Location iLoc, LocationStack &ls )
    {
        mlir::Value value;
        ParserPerFunctionState &f = lookupFunctionState( currentFuncName );
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
                i = createIndexCast( iLoc, iValue, ls );

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

    mlir::Value Builder::createIndexCast( mlir::Location loc, mlir::Value val, LocationStack &ls )
    {
        mlir::IndexType indexTy = builder.getIndexType();
        mlir::Type valTy = val.getType();

        if ( valTy == indexTy )
        {
            return val;
        }

        if ( !valTy.isSignlessInteger( 64 ) && valTy.isInteger() )
        {
            val = createCastIfNeeded( loc, val, typ.i64, ls );
            valTy = typ.i64;
        }

        // Only support i64, or castable to i64, for now
        if ( !valTy.isSignlessInteger( 64 ) )
        {
            // If it's a non-i64 IntegerType, we could cast up to i64, and then cast that to index.
            emitInternalError(
                loc, __FILE__, __LINE__, __func__,
                std::format( "NYI: createIndexCast from type {} is not supported.", mlirTypeToString( valTy ) ),
                currentFuncName );
            return mlir::Value{};
        }

        ls.push_back( loc );
        return mlir::arith::IndexCastOp::create( builder, loc, indexTy, val );
    }

    mlir::Value Builder::createCastIfNeeded( mlir::Location loc, mlir::Value value, mlir::Type desiredType,
                                             LocationStack &ls )
    {
        mlir::Value newValue{};

        // tolerate emitError codepaths where things go bad, logging an internal error and continuing
        if ( !value )
        {
            emitInternalError(
                loc, __FILE__, __LINE__, __func__,
                "NULL value in cast attempt.",
                currentFuncName );

            return value;
        }

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

    void Builder::createAssignment( mlir::Location loc, mlir::Value resultValue, const std::string &currentVarName,
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

                mlir::Value i = createIndexCast( loc, currentIndexExpr, ls );

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

    bool Builder::isDeclared( const std::string &varName )
    {
        // Get the single scope
        ParserPerFunctionState &f = lookupFunctionState( currentFuncName );

        mlir::Value v = f.searchForVariable( varName );
        return ( v != nullptr ) ? true : false;
    }

    mlir::Type Builder::getReturnType()
    {
        mlir::Type returnType{};
        if ( currentFuncName == ENTRY_SYMBOL_NAME )
        {
            returnType = typ.i32;
        }
        else
        {
            ParserPerFunctionState &f = lookupFunctionState( currentFuncName );
            mlir::func::FuncOp funcOp = f.getFuncOp();
            llvm::ArrayRef<mlir::Type> returnTypeArray = funcOp.getFunctionType().getResults();

            if ( !returnTypeArray.empty() )
            {
                returnType = returnTypeArray[0];
            }
        }

        return returnType;
    }

    void Builder::createReturn( mlir::Location loc, mlir::Value returnValue, LocationStack &ls )
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

    mlir::Value Builder::createBinaryCompare( mlir::Location loc, silly::CmpBinOpKind what, mlir::Value lhs,
                                              mlir::Value rhs, LocationStack &ls )
    {
        ls.push_back( loc );

        return silly::CmpBinOp::create( builder, loc, typ.i1, silly::CmpBinOpKindAttr::get( this->ctx, what ), lhs,
                                        rhs )
            .getResult();
    }

    mlir::Value Builder::createUnary( mlir::Location loc, mlir::Value value, UnaryOp op, LocationStack &ls )
    {
        mlir::Value v;

        switch ( op )
        {
            case UnaryOp::Negate:
            {
                // Negation
                v = silly::NegOp::create( builder, loc, value.getType(), value ).getResult();
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
                v = createBinaryCompare( loc, silly::CmpBinOpKind::Equal, value, zero, ls );
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

    void Builder::createGet( mlir::Location gloc, const std::string &varName, mlir::Location vloc, mlir::Value indexValue,
                             mlir::Location iloc, LocationStack &ls )
    {
        silly::DeclareOp declareOp = lookupDeclareForVar( gloc, varName );
        silly::varType varTy = mlir::cast<silly::varType>( declareOp.getVar().getType() );
        mlir::Type elemType = varTy.getElementType();
        mlir::DenseI64ArrayAttr shapeAttr = varTy.getShape();
        llvm::ArrayRef<int64_t> shape = shapeAttr.asArrayRef();

        mlir::Value optIndexValue{};

        if ( indexValue )
        {
            optIndexValue = createIndexCast( iloc, indexValue, ls );
        }
        else if ( !shape.empty() )
        {
            // coverage: syntax-error/get-string.silly
            emitUserError( vloc, std::format( "Attempted GET to string literal or array?" ), currentFuncName );
            return;
        }
        else
        {
            // Scalar: load the value
        }

        mlir::Value var = declareOp.getResult();

        ls.push_back( gloc );
        silly::GetOp resultValue = silly::GetOp::create( builder, gloc, elemType );

        // mlir::Location fusedLoc = ls.fuseLocations( );
        silly::AssignOp::create( builder, gloc, var, optIndexValue, resultValue );
    }

    void Builder::createImport( mlir::Location loc, mlir::Location nameLoc, const std::string &modname )
    {
        LLVM_DEBUG( { llvm::errs() << llvm::formatv( "enterImportStatement: import: {0}\n", modname ); } );

        mlir::ModuleOp importMod = sm.findMOD( modname );
        if ( !importMod )
        {
            // coverage: driver/module-not-found.silly
            emitUserError(
                nameLoc,
                std::format( "Failed to process IMPORT {}.  All module imports must be named with --imports", modname ),
                currentFuncName );
            return;
        }

        std::vector<mlir::NamedAttribute> attrs;
        attrs.push_back(
            mlir::NamedAttribute( builder.getStringAttr( "sym_visibility" ), builder.getStringAttr( "private" ) ) );

        mlir::ModuleOp mod = rmod.get();
        silly::ModuleInsertionPointGuard ip( mod, builder );

        for ( mlir::func::FuncOp srcFuncOp : importMod.getBodyRegion().getOps<mlir::func::FuncOp>() )
        {
            // only import defined functions, not decls
            if ( !srcFuncOp.isDeclaration() )
            {
                std::string funcName = srcFuncOp.getSymName().str();
                ParserPerFunctionState &f = lookupFunctionState( funcName );
                if ( !f.getFuncOp() )    // treat declarations as idempotent.
                {
                    mlir::func::FuncOp proto = mlir::func::FuncOp::create(
                        builder, srcFuncOp.getLoc(), srcFuncOp.getSymName(), srcFuncOp.getFunctionType(), attrs );

                    f.setFuncOp( proto );
                }
            }
        }
    }

    void Builder::createFunction( LocPairs locs, const std::string &funcName, bool isDeclaration, mlir::Type returnType,
                                  std::vector<mlir::Type> &paramTypes, const std::vector<std::string> &paramNames )
    {
        LLVM_DEBUG( {
            llvm::errs() << llvm::formatv( "enterFunctionStatement: startLoc: {0}, endLoc: {1}:\n",
                                           formatLocation( locs.first ), formatLocation( locs.second ) );
        } );

        assert( !currentFuncName.empty() );
        if ( currentFuncName != ENTRY_SYMBOL_NAME )
        {
            // coverage: syntax-error/nested.silly
            //
            // To support this, exitFor would have to pop an insertion point and current-function-name,
            // and we'd have to push an insertion-point/function-name instead of just assuming that
            // we started in main and will return to there.
            emitUserError( locs.first, std::format( "Nested functions are not currently supported." ),
                           currentFuncName );
            return;
        }

        ParserPerFunctionState &f = lookupFunctionState( funcName );
        mlir::func::FuncOp funcOp = f.getFuncOp();

        if ( funcOp )
        {
            if ( !funcOp.isDeclaration() )
            {
                // coverage: syntax-error/function-redefine.silly
                emitUserError( locs.first, std::format( "Attempt to define function {} more than once", funcName ),
                               currentFuncName );
                return;
            }

            if ( isDeclaration )
            {
                // coverage: syntax-error/function-redeclare.silly
                emitUserError( locs.first, std::format( "Attempt to declare function {} more than once", funcName ),
                               currentFuncName );
                return;
            }

            mainIP = builder.saveInsertionPoint();

            // fall through to createNewFunctionState, which will set the IP.
        }
        else
        {
            mainIP = builder.saveInsertionPoint();

            builder.setInsertionPointToStart( rmod->getBody() );

            std::vector<mlir::Type> returns;
            if ( returnType )
            {
                returns.push_back( returnType );
            }

            std::vector<mlir::NamedAttribute> attrs;
            // not sure exactly why I need private.  If I omit it (for extern functions), the lowering
            // to LLVM-IR ends up the same.
            //
            // However, in simple/external/callext.silly that resulted in a verifier error,
            // stating that public was not allowed.  For now, just use private always.
            attrs.push_back(
                mlir::NamedAttribute( builder.getStringAttr( "sym_visibility" ), builder.getStringAttr( "private" ) ) );

            mlir::FunctionType funcType = builder.getFunctionType( paramTypes, returns );

            llvm::SmallVector<mlir::Location, 2> funcLocs{ locs.first, locs.second };
            mlir::Location fLoc = builder.getFusedLoc( funcLocs );

            funcOp = mlir::func::FuncOp::create( builder, fLoc, funcName, funcType, attrs );
        }

        if ( isDeclaration )
        {
            currentFuncName = funcName;
            f.setFuncOp( funcOp );
        }
        else
        {
            createNewFunctionState( locs.first, funcOp, funcName, paramNames );
        }
    }

    void Builder::finishFunction()
    {
        builder.restoreInsertionPoint( mainIP );

        currentFuncName = ENTRY_SYMBOL_NAME;
    }

    mlir::Value Builder::createCall( mlir::Location loc, const std::string &funcName, mlir::func::FuncOp funcOp,
                                     mlir::FunctionType funcType, bool callStatement,
                                     std::vector<mlir::Value> &parameters, LocationStack &ls )
    {
        mlir::Value ret{};
        if ( !funcOp )
        {
            emitInternalError( loc, __FILE__, __LINE__, __func__, std::format( "no FuncOp found for {}", funcName ),
                               currentFuncName );
            return ret;
        }

        size_t psz = parameters.size();
        size_t fsz = funcType.getInputs().size();
        if ( psz != fsz )
        {
            // coverage: syntax-error/call-wrong-params.silly
            emitUserError(
                loc,
                std::format( "Mismatched number of arguments to call of function {}.  Passing: {}, required: {}.",
                             funcName, psz, fsz ),
                currentFuncName );
            return ret;
        }

        mlir::TypeRange resultTypes = funcType.getResults();

        mlir::func::CallOp callOp;
        if ( callStatement )
        {
            // mlir::Location fusedLoc = ls.fuseLocations( );
            callOp = mlir::func::CallOp::create( builder, loc, resultTypes, funcName, parameters );
        }
        else
        {
            ls.push_back( loc );
            callOp = mlir::func::CallOp::create( builder, loc, resultTypes, funcName, parameters );
        }

        // Return the first result (or null for void calls)
        if ( !resultTypes.empty() )
        {
            ret = callOp.getResults()[0];
        }

        return ret;
    }

    void Builder::createFor( mlir::Location loc, const std::string &varName, mlir::Type elemType, mlir::Location varLoc,
                             mlir::Value start, mlir::Value end, mlir::Value step, LocationStack &ls )
    {
        bool declared = isDeclared( varName );
        if ( declared )
        {
            // coverage: syntax-error/shadow-induction.silly
            emitUserError( loc, std::format( "Induction variable {} clashes with declared variable", varName ),
                           currentFuncName );
            return;
        }

        ParserPerFunctionState &f = lookupFunctionState( currentFuncName );
        mlir::Value p = f.searchForInduction( varName );
        if ( p )
        {
            // coverage: syntax-error/triple-for-shadow.silly syntax-error/nested-induction-conflict.silly
            emitUserError( loc, std::format( "Induction variable {} used by enclosing FOR", varName ),
                           currentFuncName );
            return;
        }

        std::string s;

        if ( !step )
        {
            mlir::IntegerType ity = mlir::dyn_cast<mlir::IntegerType>( elemType );
            unsigned width = ity.getWidth();

            ls.push_back( loc );

            //'scf.for' op failed to verify that all of {lowerBound, upperBound, step} have same type
            step = mlir::arith::ConstantIntOp::create( builder, loc, 1, width );
            step = createCastIfNeeded( loc, step, elemType, ls );
        }

        // mlir::Location fusedLoc = ls.fuseLocations( );
        mlir::scf::ForOp forOp = mlir::scf::ForOp::create( builder, loc, start, end, step );
        f.pushToInsertionPointStack( forOp.getOperation() );

        mlir::Block &loopBody = forOp.getRegion().front();
        mlir::Value inductionVar = loopBody.getArgument( 0 );

        builder.setInsertionPointToStart( &loopBody );

        f.pushInductionVariable( varName, inductionVar );

        silly::DebugNameOp::create( builder, varLoc, inductionVar, varName, mlir::Value{} );
    }

    void Builder::finishFor( mlir::Location loc )
    {
        ParserPerFunctionState &f = lookupFunctionState( currentFuncName );
        if ( f.haveInsertionPointStack() )
        {
            f.popFromInsertionPointStack( builder );
            f.popInductionVariable();
        }
        else
        {
            emitInternalError( loc, __FILE__, __LINE__, __func__, "empty insertionPointStack", currentFuncName );
        }
    }

    void Builder::selectElseBlock( mlir::Location loc )
    {
        mlir::Block *currentBlock = builder.getInsertionBlock();
        assert( currentBlock );

        // Get the parent region of the current block (the then region).
        mlir::Region *parentRegion = currentBlock->getParent();

        // Verify it's inside an scf.if by checking the parent op.
        mlir::Operation *parentOp = parentRegion->getParentOp();
        mlir::scf::IfOp ifOp = dyn_cast<mlir::scf::IfOp>( parentOp );

        if ( !ifOp )
        {
            emitInternalError( loc, __FILE__, __LINE__, __func__,
                               "Current insertion point must be inside an scf.if then region", currentFuncName );
            return;
        }

        // Set the insertion point to the start of the else region's (first) block.
        mlir::Region &elseRegion = ifOp.getElseRegion();
        mlir::Block &elseBlock = elseRegion.front();
        builder.setInsertionPointToStart( &elseBlock );
    }

    void Builder::finishIfElifElse()
    {
        ParserPerFunctionState &f = lookupFunctionState( currentFuncName );
        f.popFromInsertionPointStack( builder );
    }

    void Builder::createIf( mlir::Location loc, mlir::Value conditionPredicate, bool saveIP, LocationStack &ls )
    {
        // mlir::Location fusedLoc = ls.fuseLocations( );
        mlir::scf::IfOp ifOp = mlir::scf::IfOp::create( builder, loc, conditionPredicate,
                                                        /*withElseRegion=*/true );

        if ( saveIP )
        {
            ParserPerFunctionState &f = lookupFunctionState( currentFuncName );
            f.pushToInsertionPointStack( ifOp.getOperation() );
        }

        mlir::Block &thenBlock = ifOp.getThenRegion().front();
        builder.setInsertionPointToStart( &thenBlock );
    }

    void Builder::enterScopedRegion( mlir::Location loc, bool wantScope )
    {
        ParserPerFunctionState &f = lookupFunctionState( currentFuncName );

        mlir::Value value{};

        // not doing this right now for FUNCTION
        if ( wantScope )
        {
            value = silly::DebugScopeOp::create( builder, loc, typ.i1 ).getResult();
        }

        // keep stack balanced, signals function scope when !isFunctionBody
        f.startScope( value );
    }

    void Builder::exitScopedRegion()
    {
        ParserPerFunctionState &f = lookupFunctionState( currentFuncName );
        f.endScope();
    }

    void Builder::createStringDeclare( mlir::Location loc, const std::string &varName, mlir::Location aloc,
                                       const std::string &arrayBoundsString, bool haveInit, const std::string &strLit,
                                       LocationStack &ls )
    {
        std::vector<mlir::Value> initializers;
        createDeclaration( loc, varName, typ.i8, aloc, arrayBoundsString, false, initializers, ls );

        if ( haveInit )
        {
            silly::DeclareOp declareOp = lookupDeclareForVar( loc, varName );
            mlir::Value var = declareOp.getResult();

            silly::StringLiteralOp stringLiteral = createStringLiteral( loc, strLit, ls );
            if ( stringLiteral )
            {
                mlir::Value i{};

                // mlir::Location fusedLoc = ls.fuseLocations( );
                silly::AssignOp::create( builder, loc, var, i, stringLiteral );
            }
        }
    }
}    // namespace silly

// vim: et ts=4 sw=4
