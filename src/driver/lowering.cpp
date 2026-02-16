///
/// @file    lowering.cpp
/// @author  Peeter Joot <peeterjoot@pm.me>
/// @brief   This file implements the LLVM-IR lowering pattern matching operators
///
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/Module.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/TargetParser/Host.h>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h>
#include <mlir/Conversion/LLVMCommon/Pattern.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/LLVMIR/LLVMAttrs.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

#include <format>
#include <numeric>

#include "DriverState.hpp"
#include "LoweringContext.hpp"
#include "createSillyToLLVMLoweringPass.hpp"
#include "helper.hpp"

/// --debug- type for lowering
#define DEBUG_TYPE "silly-lowering"

namespace silly
{
    /// Lower silly::DeclareOp
    class DeclareOpLowering : public mlir::ConversionPattern
    {
       private:
        LoweringContext& lState;    ///< lowering context (including DriverState)

       public:
        /// Constructor boilerplate for DeclareOpLowering
        DeclareOpLowering( LoweringContext& loweringState, mlir::MLIRContext* context, mlir::PatternBenefit benefit )
            : mlir::ConversionPattern( silly::DeclareOp::getOperationName(), benefit, context ), lState( loweringState )
        {
        }

        /// Lowering workhorse for silly::DeclareOp
        mlir::LogicalResult matchAndRewrite( mlir::Operation* op, mlir::ArrayRef<mlir::Value> operands,
                                             mlir::ConversionPatternRewriter& rewriter ) const override
        {
            silly::DeclareOp declareOp = cast<silly::DeclareOp>( op );
            mlir::Location loc = declareOp.getLoc();

            LLVM_DEBUG( llvm::dbgs() << "Lowering silly.declare: " << declareOp << '\n' );

            rewriter.setInsertionPoint( declareOp );

            silly::varType varTy = mlir::cast<silly::varType>( declareOp.getVar().getType() );
            mlir::Type elemType = varTy.getElementType();

            if ( !elemType.isIntOrFloat() )
            {
                return rewriter.notifyMatchFailure( declareOp, "declare type must be integer or float" );
            }

            unsigned elemSizeInBits = elemType.getIntOrFloatBitWidth();
            unsigned elemSizeInBytes = ( elemSizeInBits + 7 ) / 8;

            // FIXME: could pack array creation for i1 types (elemType.isInteger( 1 )).  For now, just use a
            // separate byte for each.
            unsigned alignment = lState.preferredTypeAlignment( declareOp, elemType );

            mlir::DenseI64ArrayAttr shapeAttr = varTy.getShape();
            llvm::ArrayRef<int64_t> shape = shapeAttr.asArrayRef();

            mlir::Value sizeVal;
            mlir::Value bytesVal;
            int64_t arraySize = 1;
            if ( !shape.empty() )
            {
                arraySize = shape[0];

                if ( ( arraySize <= 0 ) || ( shape.size() != 1 ) )
                {
                    return rewriter.notifyMatchFailure(
                        declareOp, llvm::formatv( "expected non-zero arraySize ({0}), and one varType dimension ({1})",
                                                  arraySize, shape.size() ) );
                }

                sizeVal = rewriter.create<mlir::LLVM::ConstantOp>( loc, lState.typ.i64,
                                                                   rewriter.getI64IntegerAttr( arraySize ) );
                bytesVal = rewriter.create<mlir::LLVM::ConstantOp>(
                    loc, lState.typ.i64, rewriter.getI64IntegerAttr( arraySize * elemSizeInBytes ) );
            }
            else
            {
                sizeVal = lState.getI64one( loc, rewriter );
                bytesVal = rewriter.create<mlir::LLVM::ConstantOp>( loc, lState.typ.i64,
                                                                    rewriter.getI64IntegerAttr( elemSizeInBytes ) );
            }

            mlir::LLVM::AllocaOp allocaOp =
                rewriter.create<mlir::LLVM::AllocaOp>( loc, lState.typ.ptr, elemType, sizeVal, alignment );

            auto init = declareOp.getInitializers();
            if ( init.size() )
            {
                mlir::Type elemType = allocaOp.getElemType();
                unsigned alignment = lState.preferredTypeAlignment( declareOp, elemType );

                if ( !shape.empty() )
                {
                    for ( size_t i = 0; i < init.size(); ++i )
                    {
                        mlir::Value iVal64 = rewriter.create<mlir::LLVM::ConstantOp>(
                            loc, lState.typ.i64, rewriter.getI64IntegerAttr( static_cast<int64_t>( i ) ) );

                        mlir::IndexType indexTy = rewriter.getIndexType();
                        mlir::Value idxIndex = rewriter.create<mlir::arith::IndexCastOp>( loc, indexTy, iVal64 );

                        if ( mlir::failed( lState.generateAssignment(
                                 loc, rewriter, declareOp, init[i], elemType, allocaOp, alignment,
                                 mlir::cast<mlir::TypedValue<mlir::IndexType>>( idxIndex ) ) ) )
                        {
                            return mlir::failure();
                        }
                    }
                }
                else
                {
                    if ( init.size() != 1 )
                    {
                        return rewriter.notifyMatchFailure(
                            declareOp, llvm::formatv( "scalar initializer count: {0} is not one.", init.size() ) );
                    }

                    if ( mlir::failed( lState.generateAssignment( loc, rewriter, declareOp, init[0], elemType, allocaOp,
                                                                  alignment, mlir::TypedValue<mlir::IndexType>{} ) ) )
                    {
                        return mlir::failure();
                    }
                }
            }
            else
            {
                lState.insertFill( loc, rewriter, allocaOp, bytesVal );
            }

            std::string funcName = lookupFuncNameForOp( declareOp );
            lState.setAlloca( funcName, declareOp.getOperation(), allocaOp.getOperation() );

            // rewriter.replaceOp( declareOp, allocaOp.getResult() ); // this failed.  typeconverter didn't work as
            // hoped.
            rewriter.eraseOp( declareOp );

            return mlir::success();
        }
    };

    /// Lower silly::StringLiteralOp
    class StringLiteralOpLowering : public mlir::ConversionPattern
    {
       private:
        LoweringContext& lState;    ///< lowering context (including DriverState)

       public:
        /// Constructor boilerplate for StringLiteralOpLowering
        StringLiteralOpLowering( LoweringContext& loweringState, mlir::MLIRContext* context,
                                 mlir::PatternBenefit benefit )
            : mlir::ConversionPattern( silly::StringLiteralOp::getOperationName(), benefit, context ),
              lState( loweringState )
        {
        }

        /// Lowering workhorse for silly::StringLiteralOp
        mlir::LogicalResult matchAndRewrite( mlir::Operation* op, mlir::ArrayRef<mlir::Value> operands,
                                             mlir::ConversionPatternRewriter& rewriter ) const override
        {
            silly::StringLiteralOp stringLiteralOp = cast<silly::StringLiteralOp>( op );
            mlir::Location loc = stringLiteralOp.getLoc();

            mlir::StringAttr strAttr = stringLiteralOp.getValueAttr();
            std::string strValue = strAttr.getValue().str();
            size_t strLen = strValue.size();

            mlir::LLVM::GlobalOp globalOp = lState.lookupOrInsertGlobalOp( loc, rewriter, strAttr, strLen );
            if ( !globalOp )
            {
                return rewriter.notifyMatchFailure( op, "Failed to create or lookup string literal global" );
            }

            rewriter.eraseOp( op );
            return mlir::success();
        }
    };

    /// Lower silly::AssignOp to llvm.store (after type conversions, if required)
    class AssignOpLowering : public mlir::ConversionPattern
    {
       private:
        LoweringContext& lState;    ///< lowering context (including DriverState)

       public:
        /// Constructor boilerplate for AssignOpLowering
        AssignOpLowering( LoweringContext& loweringState, mlir::MLIRContext* context, mlir::PatternBenefit benefit )
            : mlir::ConversionPattern( silly::AssignOp::getOperationName(), benefit, context ), lState{ loweringState }
        {
        }

        /// Lowering workhorse for silly::AssignOp
        mlir::LogicalResult matchAndRewrite( mlir::Operation* op, mlir::ArrayRef<mlir::Value> operands,
                                             mlir::ConversionPatternRewriter& rewriter ) const override
        {
            silly::AssignOp assignOp = cast<silly::AssignOp>( op );
            mlir::Location loc = assignOp.getLoc();

            // silly.assign %0 : <i64[[]]> = %c1_i64 : i64 loc(#loc3)
            LLVM_DEBUG( llvm::dbgs() << "Lowering AssignOp: " << *op << '\n' );

            mlir::Value var = assignOp.getVar();
            assert( var );

            std::string funcName = lookupFuncNameForOp( op );

            silly::DeclareOp declareOp = var.getDefiningOp<silly::DeclareOp>();
            mlir::LLVM::AllocaOp allocaOp = lState.getAlloca( funcName, declareOp.getOperation() );

            LLVM_DEBUG( {
                llvm::dbgs() << "AssignOp.  module state::\n";
                mlir::ModuleOp mod = op->getParentOfType<mlir::ModuleOp>();
                mod->dump();
            } );
            assert( allocaOp );

            mlir::Value value = assignOp.getValue();

            silly::varType varTy = mlir::cast<silly::varType>( var.getType() );
            mlir::Type elemType = varTy.getElementType();
            unsigned alignment = lState.preferredTypeAlignment( op, elemType );

            if ( mlir::failed( lState.generateAssignment( loc, rewriter, op, value, elemType, allocaOp, alignment,
                                                          assignOp.getIndex() ) ) )
            {
                return mlir::failure();
            }

            rewriter.eraseOp( op );
            return mlir::success();
        }
    };

    /// Lower silly::LoadOp
    class LoadOpLowering : public mlir::ConversionPattern
    {
       private:
        LoweringContext& lState;    ///< lowering context (including DriverState)

       public:
        /// Constructor boilerplate for LoadOpLowering
        LoadOpLowering( LoweringContext& loweringState, mlir::MLIRContext* context, mlir::PatternBenefit benefit )
            : mlir::ConversionPattern( silly::LoadOp::getOperationName(), benefit, context ), lState{ loweringState }
        {
        }

        /// Lowering workhorse for silly::LoadOp
        mlir::LogicalResult matchAndRewrite( mlir::Operation* op, mlir::ArrayRef<mlir::Value> operands,
                                             mlir::ConversionPatternRewriter& rewriter ) const override
        {
            silly::LoadOp loadOp = cast<silly::LoadOp>( op );
            mlir::Location loc = loadOp.getLoc();

            // %0 = silly.load "i1v" : i1
            LLVM_DEBUG( llvm::dbgs() << "Lowering silly.load: " << *op << '\n' );

            mlir::Value var = loadOp.getVar();
            assert( var );
            std::string funcName = lookupFuncNameForOp( op );
            silly::DeclareOp declareOp = var.getDefiningOp<silly::DeclareOp>();
            mlir::LLVM::AllocaOp allocaOp = lState.getAlloca( funcName, declareOp.getOperation() );
            assert( allocaOp );
            mlir::TypedValue<mlir::IndexType> optIndex = loadOp.getIndex();

            mlir::Type elemType = allocaOp.getElemType();
            mlir::Value load;

            if ( loadOp.getResult().getType() == lState.typ.ptr )
            {
                load = allocaOp.getResult();
            }
            else
            {
                if ( optIndex )
                {
                    mlir::Value indexVal = optIndex;
                    mlir::Value basePtr = allocaOp.getResult();

                    int64_t numElems = 0;
                    if ( mlir::LLVM::ConstantOp allocationBoundsConstOp =
                             allocaOp.getArraySize().getDefiningOp<mlir::LLVM::ConstantOp>() )
                    {
                        mlir::IntegerAttr intAttr =
                            mlir::dyn_cast<mlir::IntegerAttr>( allocationBoundsConstOp.getValue() );
                        numElems = intAttr.getInt();
                    }

                    if ( !numElems )
                    {
                        return rewriter.notifyMatchFailure( op, "zero elements for attempted index access" );
                    }

                    if ( mlir::arith::ConstantIndexOp constOp = indexVal.getDefiningOp<mlir::arith::ConstantIndexOp>() )
                    {
                        int64_t idx = constOp.value();
                        if ( idx < 0 || idx >= numElems )
                        {
                            return loadOp.emitError() << "static out-of-bounds array access: index " << idx
                                                      << " is out of bounds for array of size " << numElems;
                        }
                    }

                    // Cast index to i64 for LLVM GEP
                    mlir::Value idxI64 = rewriter.create<mlir::arith::IndexCastOp>( loc, lState.typ.i64, indexVal );

                    // GEP to element
                    mlir::Value elemPtr =
                        rewriter.create<mlir::LLVM::GEPOp>( loc,
                                                            basePtr.getType(),    // result type: ptr-to-elem
                                                            elemType,             // pointee type
                                                            basePtr, mlir::ValueRange{ idxI64 } );

                    // Load array element
                    load = rewriter.create<mlir::LLVM::LoadOp>( loc, elemType, elemPtr ).getResult();
                }
                else
                {
                    // Scalar load
                    load = rewriter.create<mlir::LLVM::LoadOp>( loc, elemType, allocaOp ).getResult();
                }
            }

            LLVM_DEBUG( llvm::dbgs() << "new load op: " << load << '\n' );
            rewriter.replaceOp( op, load );

            return mlir::success();
        }
    };

#if 0
    // Now unused (again)
    template <class SillOpType>
    class LowerByDeletion : public mlir::ConversionPattern
    {
       private:
        LoweringContext& lState;

       public:
        LowerByDeletion( LoweringContext& loweringState, mlir::MLIRContext* context, mlir::PatternBenefit benefit )
            : mlir::ConversionPattern( SillOpType::getOperationName(), benefit, context ), lState{ loweringState }
        {
        }

        mlir::LogicalResult matchAndRewrite( mlir::Operation* op, mlir::ArrayRef<mlir::Value> operands,
                                             mlir::ConversionPatternRewriter& rewriter ) const override
        {
            LLVM_DEBUG( llvm::dbgs() << "Lowering (by erase): " << *op << '\n' );
            rewriter.eraseOp( op );
            return mlir::success();
        }
    };
#endif

    /// Lower silly::DebugNameOp
    class DebugNameOpLowering : public mlir::ConversionPattern
    {
       private:
        LoweringContext& lState;    ///< lowering context (including DriverState)

       public:
        /// Constructor boilerplate for DebugNameOpLowering
        DebugNameOpLowering( LoweringContext& loweringState, mlir::MLIRContext* context, mlir::PatternBenefit benefit )
            : mlir::ConversionPattern( silly::DebugNameOp::getOperationName(), benefit, context ),
              lState( loweringState )
        {
        }

        /// Lowering workhorse for silly::DebugNameOp
        mlir::LogicalResult matchAndRewrite( mlir::Operation* op, mlir::ArrayRef<mlir::Value> operands,
                                             mlir::ConversionPatternRewriter& rewriter ) const override
        {
            silly::DebugNameOp debugNameOp = cast<silly::DebugNameOp>( op );
            mlir::Value value = debugNameOp.getValue();
            mlir::Location loc = debugNameOp.getLoc();
            mlir::FileLineColLoc fileLoc = locationToFLCLoc( loc );
            std::string funcName = lookupFuncNameForOp( debugNameOp );

            if ( silly::DeclareOp declareOp = value.getDefiningOp<silly::DeclareOp>() )
            {
                std::string varName = debugNameOp.getName().str();
                mlir::LLVM::AllocaOp allocaOp = lState.getAlloca( funcName, declareOp.getOperation() );
                silly::varType varTy = mlir::cast<silly::varType>( declareOp.getVar().getType() );
                mlir::Type elemType = varTy.getElementType();
                LLVM_DEBUG( llvm::dbgs() << "DebugNameOpLowering: elemType: " << elemType << '\n' );

                if ( !elemType.isIntOrFloat() )
                {
                    return rewriter.notifyMatchFailure( declareOp, "declare type must be integer or float" );
                }

                unsigned elemSizeInBits = elemType.getIntOrFloatBitWidth();

                mlir::DenseI64ArrayAttr shapeAttr = varTy.getShape();
                llvm::ArrayRef<int64_t> shape = shapeAttr.asArrayRef();

                int64_t arraySize = 1;
                if ( !shape.empty() )
                {
                    arraySize = shape[0];
                }

                if ( mlir::failed( lState.constructVariableDI( fileLoc, rewriter, declareOp, varName, elemType,
                                                               elemSizeInBits, allocaOp.getResult(), arraySize ) ) )
                {
                    return mlir::failure();
                }
            }
            else
            {
                std::string varName = debugNameOp.getName().str();

                mlir::Type elemType = value.getType();
                unsigned elemSizeInBits = elemType.getIntOrFloatBitWidth();

                if ( mlir::failed( lState.constructVariableDI( fileLoc, rewriter, debugNameOp, varName, elemType,
                                                               elemSizeInBits, value, 1 ) ) )
                {
                    return mlir::failure();
                }
            }

            rewriter.eraseOp( debugNameOp );
            return mlir::success();
        }
    };

    /// Lower silly.print (silly::PrintOp) to a call to __silly_print.
    class PrintOpLowering : public mlir::ConversionPattern
    {
       private:
        LoweringContext& lState;    ///< lowering context (including DriverState)

       public:
        /// Constructor boilerplate for PrintOpLowering
        PrintOpLowering( LoweringContext& loweringState, mlir::MLIRContext* context, mlir::PatternBenefit benefit )
            : mlir::ConversionPattern( silly::PrintOp::getOperationName(), benefit, context ), lState{ loweringState }
        {
        }

        /// Lowering workhorse for silly::PrintOp
        mlir::LogicalResult matchAndRewrite( mlir::Operation* op, mlir::ArrayRef<mlir::Value> operands,
                                             mlir::ConversionPatternRewriter& rewriter ) const override
        {
            silly::PrintOp printOp = mlir::cast<silly::PrintOp>( op );
            mlir::Location loc = printOp.getLoc();

            std::string funcName = lookupFuncNameForOp( op );
            mlir::LLVM::AllocaOp arrayAlloca = lState.getPrintArgs( funcName );
            assert( arrayAlloca );

            auto inputs = printOp.getInputs();
            size_t numArgs = inputs.size();

            int baseFlags = 0;
            if ( mlir::arith::ConstantIntOp flagOp =
                     mlir::dyn_cast<mlir::arith::ConstantIntOp>( printOp.getFlags().getDefiningOp() ) )
            {
                baseFlags = flagOp.value();
            }

            mlir::Location argLoc = loc;
            // mlir::Location argLoc = rewriter.getUnknownLoc();

            for ( size_t i = 0; i < numArgs; ++i )
            {
                bool isLast = ( i == numArgs - 1 );
                PrintFlags pf = static_cast<PrintFlags>( baseFlags );

                if ( !isLast )
                {
                    pf = static_cast<PrintFlags>( pf | static_cast<int>( silly::PrintFlags::PRINT_FLAGS_CONTINUE ) );
                }

                mlir::Value argStruct;
                if ( mlir::failed( lState.emitPrintArgStruct( argLoc, rewriter, op, inputs[i], pf, argStruct ) ) )
                {
                    return mlir::failure();
                }

                mlir::LLVM::ConstantOp indexVal =
                    rewriter.create<mlir::LLVM::ConstantOp>( argLoc, lState.typ.i64, rewriter.getI64IntegerAttr( i ) );
                mlir::LLVM::GEPOp slotPtr = rewriter.create<mlir::LLVM::GEPOp>(
                    argLoc, lState.typ.ptr, lState.printArgStructTy, arrayAlloca, mlir::ValueRange{ indexVal } );

                rewriter.create<mlir::LLVM::StoreOp>( argLoc, argStruct, slotPtr );
            }

            // Final call
            mlir::LLVM::ConstantOp numArgsConst =
                rewriter.create<mlir::LLVM::ConstantOp>( argLoc, lState.typ.i32, rewriter.getI32IntegerAttr( numArgs ) );

            rewriter.create<mlir::func::CallOp>( loc, mlir::TypeRange{}, "__silly_print",
                                                 mlir::ValueRange{ numArgsConst, arrayAlloca } );

            rewriter.eraseOp( op );
            return mlir::success();
        }
    };

    /// Lower silly::AbortOp
    class AbortOpLowering : public mlir::ConversionPattern
    {
       private:
        LoweringContext& lState;    ///< lowering context (including DriverState)

       public:
        /// Constructor boilerplate for AbortOpLowering
        AbortOpLowering( LoweringContext& loweringState, mlir::MLIRContext* context, mlir::PatternBenefit benefit )
            : mlir::ConversionPattern( silly::AbortOp::getOperationName(), benefit, context ), lState{ loweringState }
        {
        }

        /// Lowering workhorse for silly::AbortOp
        mlir::LogicalResult matchAndRewrite( mlir::Operation* op, mlir::ArrayRef<mlir::Value> operands,
                                             mlir::ConversionPatternRewriter& rewriter ) const override
        {
            silly::AbortOp abortOp = cast<silly::AbortOp>( op );
            mlir::Location loc = abortOp.getLoc();

            LLVM_DEBUG( llvm::dbgs() << "Lowering silly.abort: " << *op << '\n' );

            lState.createAbortCall( loc, rewriter );

            rewriter.eraseOp( op );

            return mlir::success();
        }
    };

    /// Lower silly::GetOp
    class GetOpLowering : public mlir::ConversionPattern
    {
       private:
        LoweringContext& lState;    ///< lowering context (including DriverState)

       public:
        /// Constructor boilerplate for GetOpLowering
        GetOpLowering( LoweringContext& loweringState, mlir::MLIRContext* context, mlir::PatternBenefit benefit )
            : mlir::ConversionPattern( silly::GetOp::getOperationName(), benefit, context ), lState{ loweringState }
        {
        }

        /// Lowering workhorse for silly::GetOp
        mlir::LogicalResult matchAndRewrite( mlir::Operation* op, mlir::ArrayRef<mlir::Value> operands,
                                             mlir::ConversionPatternRewriter& rewriter ) const override
        {
            silly::GetOp getOp = cast<silly::GetOp>( op );
            mlir::Location loc = getOp.getLoc();

            LLVM_DEBUG( llvm::dbgs() << "Lowering silly.get: " << *op << '\n' );

            mlir::Type inputType = getOp.getValue().getType();

            mlir::Value result;
            if ( mlir::failed( lState.createGetCall( loc, rewriter, op, inputType, result ) ) )
            {
                return mlir::failure();
            }

            rewriter.replaceOp( op, result );

            return mlir::success();
        }
    };

    /// Lower silly.negate (silly::NegOp) to LLVM arithmetic.
    class NegOpLowering : public mlir::ConversionPattern
    {
       private:
        LoweringContext& lState;    ///< lowering context (including DriverState)

       public:
        /// Constructor boilerplate for NegOpLowering
        NegOpLowering( LoweringContext& loweringState, mlir::MLIRContext* context, mlir::PatternBenefit benefit )
            : mlir::ConversionPattern( silly::NegOp::getOperationName(), benefit, context ), lState{ loweringState }
        {
        }

        /// Lowering workhorse for silly::NegOp
        mlir::LogicalResult matchAndRewrite( mlir::Operation* op, mlir::ArrayRef<mlir::Value> operands,
                                             mlir::ConversionPatternRewriter& rewriter ) const override
        {
            silly::NegOp negOp = cast<silly::NegOp>( op );
            mlir::Location loc = negOp.getLoc();
            mlir::Value result = operands[0];

            LLVM_DEBUG( llvm::dbgs() << "Lowering silly.negate: " << *op << '\n' );

            if ( mlir::IntegerType resulti = mlir::dyn_cast<mlir::IntegerType>( result.getType() ) )
            {
                mlir::LLVM::ConstantOp zero;
                if ( mlir::failed( lState.getIzero( loc, rewriter, op, resulti.getWidth(), zero ) ) )
                {
                    return mlir::failure();
                }

                result = rewriter.create<mlir::LLVM::SubOp>( loc, zero, result );
            }
            else if ( mlir::FloatType resultf = mlir::dyn_cast<mlir::FloatType>( result.getType() ) )
            {
                mlir::LLVM::ConstantOp zero;
                if ( resultf == lState.typ.f32 )
                {
                    zero = lState.getF32zero( loc, rewriter );
                }
                else if ( resultf == lState.typ.f64 )
                {
                    zero = lState.getF64zero( loc, rewriter );
                }
                else
                {
                    return rewriter.notifyMatchFailure(
                        op, llvm::formatv( "Unknown floating point type in negation operation lowering: {0}",
                                           result.getType() ) );
                }

                result = rewriter.create<mlir::LLVM::FSubOp>( loc, zero, result );
            }
            else
            {
                return rewriter.notifyMatchFailure(
                    op, llvm::formatv( "Unknown type in negation operation lowering: {0}", result.getType() ) );
            }

            rewriter.replaceOp( op, result );
            return mlir::success();
        }
    };

    /// type conversions and rewriter creations for numeric binary ops.
    template <class llvmIOpType, class llvmFOpType, bool allowFloat>
    mlir::LogicalResult binaryArithOpLoweringHelper( mlir::Location loc, LoweringContext& lState, mlir::Operation* op,
                                                     mlir::Value lhs, mlir::Value rhs,
                                                     mlir::ConversionPatternRewriter& rewriter, mlir::Type resultType,
                                                     bool needsLibMathIfFloat )
    {
        if ( resultType.isIntOrIndex() )
        {
            unsigned rwidth = resultType.getIntOrFloatBitWidth();

            if ( mlir::IntegerType lTyI = mlir::dyn_cast<mlir::IntegerType>( lhs.getType() ) )
            {
                unsigned width = lTyI.getWidth();

                if ( rwidth > width )
                {
                    lhs = rewriter.create<mlir::LLVM::ZExtOp>( loc, resultType, lhs );
                }
                else if ( rwidth < width )
                {
                    lhs = rewriter.create<mlir::LLVM::TruncOp>( loc, resultType, lhs );
                }
            }
            else if ( lState.isTypeFloat( lhs.getType() ) )
            {
                if ( allowFloat )
                {
                    lhs = rewriter.create<mlir::LLVM::FPToSIOp>( loc, resultType, lhs );
                }
                else
                {
                    return rewriter.notifyMatchFailure( op, "float types unsupported for integer binary operation" );
                }
            }

            if ( mlir::IntegerType rTyI = mlir::dyn_cast<mlir::IntegerType>( rhs.getType() ) )
            {
                unsigned width = rTyI.getWidth();

                if ( rwidth > width )
                {
                    rhs = rewriter.create<mlir::LLVM::ZExtOp>( loc, resultType, rhs );
                }
                else if ( rwidth < width )
                {
                    rhs = rewriter.create<mlir::LLVM::TruncOp>( loc, resultType, rhs );
                }
            }
            else if ( lState.isTypeFloat( rhs.getType() ) )
            {
                if ( allowFloat )
                {
                    rhs = rewriter.create<mlir::LLVM::FPToSIOp>( loc, resultType, rhs );
                }
                else
                {
                    return rewriter.notifyMatchFailure( op, "float types unsupported for integer binary operation" );
                }
            }

            llvmIOpType result = rewriter.create<llvmIOpType>( loc, lhs, rhs );
            rewriter.replaceOp( op, result );
        }
        else if ( allowFloat )
        {
            // Floating-point addition: ensure both operands are f64.
            if ( mlir::IntegerType lTyI = mlir::dyn_cast<mlir::IntegerType>( lhs.getType() ) )
            {
                unsigned width = lTyI.getWidth();

                if ( width == 1 )
                {
                    lhs = rewriter.create<mlir::LLVM::UIToFPOp>( loc, resultType, lhs );
                }
                else
                {
                    lhs = rewriter.create<mlir::LLVM::SIToFPOp>( loc, resultType, lhs );
                }
            }
            if ( mlir::IntegerType rTyI = mlir::dyn_cast<mlir::IntegerType>( rhs.getType() ) )
            {
                unsigned width = rTyI.getWidth();

                if ( width == 1 )
                {
                    rhs = rewriter.create<mlir::LLVM::UIToFPOp>( loc, resultType, rhs );
                }
                else
                {
                    rhs = rewriter.create<mlir::LLVM::SIToFPOp>( loc, resultType, rhs );
                }
            }

            if ( needsLibMathIfFloat )
            {
                lState.markMathLibRequired();
            }

            llvmFOpType result = rewriter.create<llvmFOpType>( loc, lhs, rhs );
            rewriter.replaceOp( op, result );
        }
        else
        {
            return rewriter.notifyMatchFailure( op, "float types unsupported for integer binary operation" );
        }

        return mlir::success();
    }

    /// Lower silly::ArithBinOp
    class ArithBinOpLowering : public mlir::ConversionPattern
    {
        LoweringContext& lState;    ///< lowering context (including DriverState)

       public:
        /// Constructor boilerplate for ArithBinOpLowering
        ArithBinOpLowering( LoweringContext& state, mlir::MLIRContext* context, mlir::PatternBenefit benefit )
            : ConversionPattern( silly::ArithBinOp::getOperationName(), benefit, context ), lState( state )
        {
        }

        /// Lowering workhorse for silly::ArithBinOp
        mlir::LogicalResult matchAndRewrite( mlir::Operation* op, mlir::ArrayRef<mlir::Value> operands,
                                             mlir::ConversionPatternRewriter& rewriter ) const override
        {
            silly::ArithBinOp binaryOp = cast<silly::ArithBinOp>( op );
            silly::ArithBinOpKind kind = binaryOp.getKind();

            mlir::Location loc = binaryOp.getLoc();

            LLVM_DEBUG( llvm::dbgs() << "Lowering silly.binary: " << *op << '\n' );

            mlir::Value lhs = operands[0];
            mlir::Value rhs = operands[1];
            mlir::Type resultType = binaryOp.getResult().getType();

            switch ( kind )
            {
                case silly::ArithBinOpKind::Add:
                {
                    return binaryArithOpLoweringHelper<mlir::LLVM::AddOp, mlir::LLVM::FAddOp, true>(
                        loc, lState, op, lhs, rhs, rewriter, resultType, false );
                }
                case silly::ArithBinOpKind::Sub:
                {
                    return binaryArithOpLoweringHelper<mlir::LLVM::SubOp, mlir::LLVM::FSubOp, true>(
                        loc, lState, op, lhs, rhs, rewriter, resultType, false );
                }
                case silly::ArithBinOpKind::Mul:
                {
                    return binaryArithOpLoweringHelper<mlir::LLVM::MulOp, mlir::LLVM::FMulOp, true>(
                        loc, lState, op, lhs, rhs, rewriter, resultType, false );
                }

                case silly::ArithBinOpKind::Div:
                {
                    return binaryArithOpLoweringHelper<mlir::LLVM::SDivOp, mlir::LLVM::FDivOp, true>(
                        loc, lState, op, lhs, rhs, rewriter, resultType, false );
                }

                case silly::ArithBinOpKind::Mod:
                {
                    // LLVM_DEBUG( llvm::dbgs() << "Lowering mod: " << *op << ' ' << lhs << ' ' << rhs << '\n' );
                    return binaryArithOpLoweringHelper<mlir::LLVM::SRemOp, mlir::LLVM::FRemOp, true>(
                        loc, lState, op, lhs, rhs, rewriter, resultType, true );
                }

                // mlir::LLVM::FAddOp is a dummy operation below, knowing that it will not ever be used:
                case silly::ArithBinOpKind::And:
                {
                    return binaryArithOpLoweringHelper<mlir::LLVM::AndOp, mlir::LLVM::FAddOp, false>(
                        loc, lState, op, lhs, rhs, rewriter, resultType, false );
                }

                case silly::ArithBinOpKind::Or:
                {
                    return binaryArithOpLoweringHelper<mlir::LLVM::OrOp, mlir::LLVM::FAddOp, false>(
                        loc, lState, op, lhs, rhs, rewriter, resultType, false );
                }

                case silly::ArithBinOpKind::Xor:
                {
                    return binaryArithOpLoweringHelper<mlir::LLVM::XOrOp, mlir::LLVM::FAddOp, false>(
                        loc, lState, op, lhs, rhs, rewriter, resultType, false );
                }
            }

            llvm_unreachable( "unknown arith binop kind" );

            return mlir::failure();
        }
    };

    /// A helper function for silly::ArithBinOp
    template <class IOpType, class FOpType, mlir::LLVM::ICmpPredicate ICmpPredS, mlir::LLVM::ICmpPredicate ICmpPredU,
              mlir::LLVM::FCmpPredicate FCmpPred>
    mlir::LogicalResult binaryCompareOpLoweringHelper( mlir::Location loc, LoweringContext& lState, mlir::Operation* op,
                                                       mlir::Value lhs, mlir::Value rhs,
                                                       mlir::ConversionPatternRewriter& rewriter )
    {
        mlir::IntegerType lTyI = mlir::dyn_cast<mlir::IntegerType>( lhs.getType() );
        mlir::IntegerType rTyI = mlir::dyn_cast<mlir::IntegerType>( rhs.getType() );
        mlir::FloatType lTyF = mlir::dyn_cast<mlir::FloatType>( lhs.getType() );
        mlir::FloatType rTyF = mlir::dyn_cast<mlir::FloatType>( rhs.getType() );

        if ( lTyI && rTyI )
        {
            unsigned lwidth = lTyI.getWidth();
            unsigned rwidth = rTyI.getWidth();
            mlir::LLVM::ICmpPredicate pred = ICmpPredS;

            if ( rwidth > lwidth )
            {
                if ( lwidth == 1 )
                {
                    lhs = rewriter.create<mlir::LLVM::ZExtOp>( loc, rTyI, lhs );
                }
                else
                {
                    lhs = rewriter.create<mlir::LLVM::SExtOp>( loc, rTyI, lhs );
                }
            }
            else if ( rwidth < lwidth )
            {
                if ( rwidth == 1 )
                {
                    rhs = rewriter.create<mlir::LLVM::ZExtOp>( loc, lTyI, rhs );
                }
                else
                {
                    rhs = rewriter.create<mlir::LLVM::SExtOp>( loc, lTyI, rhs );
                }
            }
            else if ( ( rwidth == lwidth ) && ( rwidth == 1 ) )
            {
                pred = ICmpPredU;
            }

            IOpType cmp = rewriter.create<IOpType>( loc, pred, lhs, rhs );
            rewriter.replaceOp( op, cmp.getResult() );
        }
        else if ( lTyF && rTyF )
        {
            unsigned lwidth = lTyF.getWidth();
            unsigned rwidth = rTyF.getWidth();

            if ( lwidth < rwidth )
            {
                lhs = rewriter.create<mlir::LLVM::FPExtOp>( loc, rTyF, lhs );
            }
            else if ( rwidth < lwidth )
            {
                rhs = rewriter.create<mlir::LLVM::FPExtOp>( loc, lTyF, rhs );
            }

            FOpType cmp = rewriter.create<FOpType>( loc, FCmpPred, lhs, rhs );
            rewriter.replaceOp( op, cmp.getResult() );
        }
        else
        {
            // convert integer type to float
            if ( lTyI && rTyF )
            {
                if ( lTyI == lState.typ.i1 )
                {
                    lhs = rewriter.create<mlir::arith::UIToFPOp>( loc, rTyF, lhs );
                }
                else
                {
                    lhs = rewriter.create<mlir::arith::SIToFPOp>( loc, rTyF, lhs );
                }
            }
            else if ( rTyI && lTyF )
            {
                if ( rTyI == lState.typ.i1 )
                {
                    rhs = rewriter.create<mlir::arith::UIToFPOp>( loc, lTyF, rhs );
                }
                else
                {
                    rhs = rewriter.create<mlir::arith::SIToFPOp>( loc, lTyF, rhs );
                }
            }
            else
            {
                return rewriter.notifyMatchFailure( op, "Unsupported type combination" );
            }

            FOpType cmp = rewriter.create<FOpType>( loc, FCmpPred, lhs, rhs );
            rewriter.replaceOp( op, cmp.getResult() );
        }

        return mlir::success();
    }

    /// Lower silly::CmpBinOp
    class CmpBinOpLowering : public mlir::ConversionPattern
    {
        LoweringContext& lState;    ///< lowering context (including DriverState)

       public:
        /// Constructor boilerplate for CmpBinOpLowering
        CmpBinOpLowering( LoweringContext& state, mlir::MLIRContext* context, mlir::PatternBenefit benefit )
            : ConversionPattern( silly::CmpBinOp::getOperationName(), benefit, context ), lState( state )
        {
        }

        /// Lowering workhorse for silly::CmpBinOp
        mlir::LogicalResult matchAndRewrite( mlir::Operation* op, mlir::ArrayRef<mlir::Value> operands,
                                             mlir::ConversionPatternRewriter& rewriter ) const override
        {
            silly::CmpBinOp binaryOp = cast<silly::CmpBinOp>( op );
            silly::CmpBinOpKind kind = binaryOp.getKind();

            mlir::Location loc = binaryOp.getLoc();

            LLVM_DEBUG( llvm::dbgs() << "Lowering silly.cmp: " << *op << '\n' );

            mlir::Value lhs = operands[0];
            mlir::Value rhs = operands[1];

            switch ( kind )
            {
                case silly::CmpBinOpKind::Less:
                {
                    return binaryCompareOpLoweringHelper<mlir::LLVM::ICmpOp, mlir::LLVM::FCmpOp,
                                                         mlir::LLVM::ICmpPredicate::slt, mlir::LLVM::ICmpPredicate::ult,
                                                         mlir::LLVM::FCmpPredicate::olt>( loc, lState, op, lhs, rhs,
                                                                                          rewriter );
                }
                case silly::CmpBinOpKind::LessEq:
                {
                    return binaryCompareOpLoweringHelper<mlir::LLVM::ICmpOp, mlir::LLVM::FCmpOp,
                                                         mlir::LLVM::ICmpPredicate::sle, mlir::LLVM::ICmpPredicate::ule,
                                                         mlir::LLVM::FCmpPredicate::ole>( loc, lState, op, lhs, rhs,
                                                                                          rewriter );
                }
                case silly::CmpBinOpKind::Equal:
                {
                    return binaryCompareOpLoweringHelper<mlir::LLVM::ICmpOp, mlir::LLVM::FCmpOp,
                                                         mlir::LLVM::ICmpPredicate::eq, mlir::LLVM::ICmpPredicate::eq,
                                                         mlir::LLVM::FCmpPredicate::oeq>( loc, lState, op, lhs, rhs,
                                                                                          rewriter );
                }

                case silly::CmpBinOpKind::NotEqual:
                {
                    return binaryCompareOpLoweringHelper<mlir::LLVM::ICmpOp, mlir::LLVM::FCmpOp,
                                                         mlir::LLVM::ICmpPredicate::ne, mlir::LLVM::ICmpPredicate::ne,
                                                         mlir::LLVM::FCmpPredicate::one>( loc, lState, op, lhs, rhs,
                                                                                          rewriter );
                }
            }

            llvm_unreachable( "unknown arith binop kind" );

            return mlir::failure();
        }
    };

    /// Orchestrate the lowering of the Silly dialect.
    ///
    /// When this is done, if successful, we will be left with LLVM mlir dialect Ops
    /// and a couple standalone mlir dialect Ops (func.func, module, ...)
    class SillyToLLVMLoweringPass
        : public mlir::PassWrapper<SillyToLLVMLoweringPass, mlir::OperationPass<mlir::ModuleOp>>
    {
       public:
        MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID( SillyToLLVMLoweringPass )

        /// lowering pass.  squirrel away the DriverState for later use.
        SillyToLLVMLoweringPass( silly::DriverState* pst ) : pDriverState{ pst }
        {
        }

        /// load dependent dialects
        void getDependentDialects( mlir::DialectRegistry& registry ) const override
        {
            registry.insert<mlir::LLVM::LLVMDialect, mlir::arith::ArithDialect, mlir::scf::SCFDialect>();
        }

        /// do the lowering
        void runOnOperation() override
        {
            mlir::ModuleOp mod = getOperation();
            LLVM_DEBUG( {
                llvm::dbgs() << "Starting SillyToLLVMLoweringPass on:\n";
                mod->dump();
            } );

            LoweringContext lState( mod, *pDriverState );
            lState.createDICompileUnit();

            for ( mlir::func::FuncOp funcOp : mod.getBodyRegion().getOps<mlir::func::FuncOp>() )
            {
                LLVM_DEBUG( {
                    llvm::dbgs() << "Generating !DISubroutineType() for mlir::func::FuncOp: " << funcOp.getSymName()
                                 << "\n";
                } );

                bool error = lState.createPerFuncState( funcOp );
                if ( error )
                {
                    LLVM_DEBUG( llvm::dbgs()
                                << "!DISubroutineType() creation for " << funcOp.getSymName() << " failed" );
                    signalPassFailure();
                    return;
                }
            }

            LLVM_DEBUG( {
                llvm::dbgs() << "After createPerFuncState:\n";
                mod->dump();
            } );

            // No longer two phase lowering... just one:
            {
                mlir::ConversionTarget target( getContext() );
                target.addLegalDialect<mlir::LLVM::LLVMDialect>();
                target.addIllegalOp<silly::AssignOp, silly::DeclareOp, silly::LoadOp, silly::NegOp, silly::PrintOp,
                                    silly::GetOp, silly::StringLiteralOp, silly::AbortOp, silly::DebugNameOp,
                                    silly::ArithBinOp, silly::CmpBinOp>();
                target.addLegalOp<mlir::ModuleOp, mlir::func::FuncOp, mlir::func::CallOp, mlir::func::ReturnOp>();

                target.addIllegalDialect<mlir::scf::SCFDialect>();
                target.addIllegalDialect<mlir::cf::ControlFlowDialect>();    // forces lowering

                mlir::RewritePatternSet patterns( &getContext() );
                patterns.add<AssignOpLowering, LoadOpLowering, NegOpLowering, PrintOpLowering, AbortOpLowering,
                             GetOpLowering, StringLiteralOpLowering, ArithBinOpLowering, CmpBinOpLowering,
                             DeclareOpLowering, DebugNameOpLowering>( lState, &getContext(), 1 );

                // SCF -> CF
                mlir::populateSCFToControlFlowConversionPatterns( patterns );

                mlir::arith::populateArithToLLVMConversionPatterns( lState.getTypeConverter(), patterns );

                // CF -> LLVM
                mlir::cf::populateControlFlowToLLVMConversionPatterns( lState.getTypeConverter(), patterns );

                if ( failed( applyFullConversion( mod, target, std::move( patterns ) ) ) )
                {
                    LLVM_DEBUG( llvm::dbgs() << "Silly Lowering failed\n" );
                    signalPassFailure();
                    return;
                }

                LLVM_DEBUG( {
                    llvm::dbgs() << "After silly ops lowered:\n";
                    mod->dump();
                } );
            }

            LLVM_DEBUG( {
                llvm::dbgs() << "After successful SillyToLLVMLoweringPass:\n";
                for ( mlir::Operation& op : mod->getRegion( 0 ).front() )
                {
                    op.dump();
                }
            } );
        }

       private:
        silly::DriverState*
            pDriverState;    ///< stuff from the driver (is debug enabled, ...)  Also mark when -lm will be required.
    };

}    // namespace silly

namespace mlir
{
    /// Silly dialect pass framework
    std::unique_ptr<Pass> createSillyToLLVMLoweringPass()
    {
        return createSillyToLLVMLoweringPass( nullptr );    // Default to no optimization
    }

    /// Silly dialect pass framework
    std::unique_ptr<Pass> createSillyToLLVMLoweringPass( silly::DriverState* pst )
    {
        return std::make_unique<silly::SillyToLLVMLoweringPass>( pst );
    }

    /// Custom registration with bool parameter
    void registerSillyToLLVMLoweringPass( silly::DriverState* pst )
    {
        ::mlir::registerPass( [pst]() -> std::unique_ptr<::mlir::Pass>
                              { return mlir::createSillyToLLVMLoweringPass( pst ); } );
    }
}    // namespace mlir

// vim: et ts=4 sw=4
