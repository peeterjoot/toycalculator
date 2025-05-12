/**
 * @file    lowering.cpp
 * @author  Peeter Joot <peeterjoot@pm.me>
 * @brief   This file implements the LLVM-IR lowering pattern matching operators
 */
#include <llvm/ADT/StringRef.h>
#include <llvm/BinaryFormat/Dwarf.h>    // For DW_LANG_C, DW_ATE_*
#include <llvm/Support/Debug.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/LLVMCommon/Pattern.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>    // For future multi-function support.
#include <mlir/Dialect/LLVMIR/LLVMAttrs.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Location.h>    // For FileLineColLoc
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

#include <numeric>

#include "ToyDialect.h"
#include "lowering.h"

#define DEBUG_TYPE "toy-lowering"

using namespace mlir;

namespace toy
{
    //
    // This structure was added initially because my LLVM IR dump was not showing the normal
    // DI info that gets translated to DWARF.
    //
    // I suspect that this is because there's a new way of doing this, described
    // in:
    //    https://llvm.org/docs/RemoveDIsDebugInfo.html
    //
    // This new way is hidden by default.  Why I don't get DWARF in the end is
    // probably a different issue?
    //
    // ... but now that I have something that I can pass around between my lowering operator matching classes, it's
    // useful for other stuff:
    class loweringContext
    {
       public:
        DenseMap<llvm::StringRef, mlir::LLVM::AllocaOp> symbolToAlloca;
        mlir::LLVM::LLVMFuncOp mainFunc;
        mlir::LLVM::LLVMFuncOp printFunc;

        mlir::LLVM::ConstantOp getI64one( mlir::Location loc, ConversionPatternRewriter& rewriter )
        {
            if ( !c_one_I64 )
            {
                one_I64 =
                    rewriter.create<LLVM::ConstantOp>( loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr( 1 ) );
                c_one_I64 = true;
            }

            return one_I64;
        }

        mlir::LLVM::ConstantOp getI32zero( mlir::Location loc, ConversionPatternRewriter& rewriter )
        {
            if ( !c_zero_I32 )
            {
                zero_I32 =
                    rewriter.create<LLVM::ConstantOp>( loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr( 0 ) );
                c_zero_I32 = true;
            }

            return zero_I32;
        }

        mlir::LLVM::ConstantOp getF64zero( mlir::Location loc, ConversionPatternRewriter& rewriter )
        {
            if ( !c_zero_F64 )
            {
                zero_F64 =
                    rewriter.create<LLVM::ConstantOp>( loc, rewriter.getF64Type(), rewriter.getF64FloatAttr( 1 ) );
                c_zero_F64 = true;
            }

            return zero_F64;
        }

       private:
        // caching these may not be a good idea, as they are created with a single loc value.
        mlir::LLVM::ConstantOp one_I64;
        mlir::LLVM::ConstantOp zero_F64;
        mlir::LLVM::ConstantOp zero_I32;
        bool c_one_I64{};
        bool c_zero_F64{};
        bool c_zero_I32{};
    };

#if 0
    mlir::FileLineColLoc getLocation( mlir::Location loc )
    {
        // Cast Location to FileLineColLoc
        auto fileLineLoc = mlir::dyn_cast<mlir::FileLineColLoc>( loc );
        if ( !fileLineLoc )
        {
            throw exception_with_context( __FILE__, __LINE__, __func__, "Internal error: Expected only FileLineColLoc Location info." );
        }

        return fileLineLoc;
    }
#endif

    // Lower toy.program to an LLVM function.
    class ProgramOpLowering : public ConversionPattern
    {
       private:
        loweringContext& lState;

       public:
        ProgramOpLowering( loweringContext& lState_, MLIRContext* context )
            : ConversionPattern( toy::ProgramOp::getOperationName(), 1, context ), lState{ lState_ }
        {
        }

        LogicalResult matchAndRewrite( Operation* op, ArrayRef<Value> operands,
                                       ConversionPatternRewriter& rewriter ) const override
        {
            auto programOp = cast<toy::ProgramOp>( op );
            auto loc = programOp.getLoc();

            LLVM_DEBUG( llvm::dbgs() << "Lowering toy.program: " << *op << '\n' << loc << '\n' );

            // Create an entry block in the function
            Block* entryBlock = lState.mainFunc.addEntryBlock( rewriter );
            rewriter.setInsertionPointToStart( entryBlock );

            // Inline the toy.program's region into the function's entry block
            Region& programRegion = programOp.getRegion();
            if ( !programRegion.hasOneBlock() )
            {
                return rewriter.notifyMatchFailure( programOp, "toy.program must have exactly one block" );
            }

            // Move the block's operations (e.g., toy.return) into the entry
            // block
            Block& programBlock = programRegion.front();
            rewriter.inlineRegionBefore( programRegion, entryBlock );
            rewriter.mergeBlocks( &programBlock, entryBlock,
                                  /* argValues */ {} );

            // Erase the original program op
            rewriter.eraseOp( op );
            // LLVM_DEBUG(llvm::dbgs() << "IR after erasing toy.program:\n" << *op->getParentOp() << "\n");

            // Recursively convert the inlined operations (e.g., toy.return)
            return success();
        }
    };

    class DeclareOpLowering : public ConversionPattern
    {
       private:
        loweringContext& lState;

       public:
        DeclareOpLowering( loweringContext& lState_, MLIRContext* context )
            : ConversionPattern( toy::DeclareOp::getOperationName(), 1, context ), lState{ lState_ }
        {
        }

        LogicalResult matchAndRewrite( Operation* op, ArrayRef<Value> operands,
                                       ConversionPatternRewriter& rewriter ) const override
        {
            auto declareOp = cast<toy::DeclareOp>( op );
            auto loc = declareOp.getLoc();

            //   toy.declare "x" : i32
            LLVM_DEBUG( llvm::dbgs() << "Lowering toy.declare: " << declareOp << '\n' );

            auto varName = declareOp.getName();
            auto elemType = declareOp.getType();
            int64_t numElements = 1;    // scalar only for now.

            if ( !elemType.isIntOrFloat() )
            {
                return rewriter.notifyMatchFailure( declareOp, "declare type must be integer or float" );
            }

            unsigned elemSizeInBits = elemType.getIntOrFloatBitWidth();
            unsigned elemSizeInBytes = ( elemSizeInBits + 7 ) / 8;
            int64_t totalSizeInBytes = numElements * elemSizeInBytes;

            auto ptrType = LLVM::LLVMPointerType::get( rewriter.getContext() );
            auto one = lState.getI64one( loc, rewriter );
            auto module = op->getParentOfType<ModuleOp>();
            if ( !module )
            {
                return rewriter.notifyMatchFailure( declareOp, "declare op must be inside a module" );
            }

            mlir::DataLayout dataLayout( module );
            unsigned alignment = dataLayout.getTypePreferredAlignment( elemType );
            assert( alignment == totalSizeInBytes );    // only simple fixed size types are currently supported (no
                                                        // arrays, or structures.)

            auto newAllocaOp = rewriter.create<LLVM::AllocaOp>( loc, ptrType, elemType, one, alignment );

            lState.symbolToAlloca[varName] = newAllocaOp;

            // Erase the declare op
            rewriter.eraseOp( op );
            // LLVM_DEBUG(llvm::dbgs() << "IR after erasing toy.declare:\n" << *op->getParentOp() << "\n");

            return success();
        }
    };

    // Lower AssignOp to llvm.store (after type conversions, if required)
    class AssignOpLowering : public ConversionPattern
    {
       private:
        loweringContext& lState;

       public:
        AssignOpLowering( loweringContext& lState_, MLIRContext* ctx )
            : ConversionPattern( toy::AssignOp::getOperationName(), 1, ctx ), lState{ lState_ }
        {
        }

        LogicalResult matchAndRewrite( Operation* op, ArrayRef<Value> operands,
                                       ConversionPatternRewriter& rewriter ) const override
        {
            auto assignOp = cast<toy::AssignOp>( op );
            auto loc = assignOp.getLoc();

            // (ins StrAttr:$name, AnyType:$value);
            // toy.assign "x", %0 : i32
            LLVM_DEBUG( llvm::dbgs() << "Lowering AssignOp: " << *op << '\n' );

            auto name = assignOp.getName();
            auto value = assignOp.getValue();
            auto valType = value.getType();
            auto allocaOp = lState.symbolToAlloca[name];

            // name: i1v
            // value: %true = arith.constant true
            // valType: i1
            LLVM_DEBUG( llvm::dbgs() << "name: " << name << '\n' );
            LLVM_DEBUG( llvm::dbgs() << "value: " << value << '\n' );
            LLVM_DEBUG( llvm::dbgs() << "valType: " << valType << '\n' );
            // allocaOp.dump(); // %1 = llvm.alloca %0 x i1 {alignment = 1 : i64} : (i64) -> !llvm.ptr

            // auto memType = allocaOp.getType(); // llvm.ptr

            // extract parameters from the allocaOp so we know what to do here:
            Type elemType = allocaOp.getElemType();
            // Value alignment = allocaOp.getAlignment();
            if ( auto constOp = allocaOp.getArraySize().getDefiningOp<LLVM::ConstantOp>() )
            {
                if ( auto intAttr = mlir::dyn_cast<IntegerAttr>( constOp.getValue() ) )
                {
                    int64_t numElems = intAttr.getInt();

                    assert( numElems == 1 );
                }
                else
                {
                    assert( 0 );    // shouldn't happen.
                }
            }

            // LLVM_DEBUG( llvm::dbgs() << "memType: " << memType << '\n' );
            LLVM_DEBUG( llvm::dbgs() << "elemType: " << elemType << '\n' );
            // LLVM_DEBUG( llvm::dbgs() << "elemType: " << elemType << '\n' );

            if ( mlir::isa<mlir::Float64Type>( valType ) )
            {
                if ( mlir::isa<mlir::IntegerType>( elemType ) )
                {
                    value = rewriter.create<LLVM::FPToSIOp>( loc, elemType, value );
                }
                else if ( mlir::isa<mlir::Float32Type>( elemType ) )
                {
                    value = rewriter.create<LLVM::FPTruncOp>( loc, elemType, value );
                }
            }
            else if ( mlir::isa<mlir::Float32Type>( valType ) )
            {
                if ( mlir::isa<mlir::IntegerType>( elemType ) )
                {
                    value = rewriter.create<LLVM::FPToSIOp>( loc, elemType, value );
                }
                else if ( mlir::isa<mlir::Float64Type>( elemType ) )
                {
                    value = rewriter.create<LLVM::FPExtOp>( loc, elemType, value );
                }
            }
            else if ( auto viType = mlir::cast<mlir::IntegerType>( valType ) )
            {
                auto vwidth = viType.getWidth();

                if ( mlir::isa<mlir::Float64Type>( elemType ) || mlir::isa<mlir::Float32Type>( elemType ) )
                {
                    if ( vwidth == 1 )
                    {
                        value = rewriter.create<LLVM::UIToFPOp>( loc, elemType, value );
                    }
                    else
                    {
                        value = rewriter.create<LLVM::SIToFPOp>( loc, elemType, value );
                    }
                }
                else
                {
                    auto miType = mlir::cast<mlir::IntegerType>( elemType );

                    auto mwidth = miType.getWidth();
                    if ( vwidth > mwidth )
                    {
                        value = rewriter.create<mlir::LLVM::TruncOp>( loc, elemType, value );
                    }
                    else if ( vwidth < mwidth )
                    {
                        value = rewriter.create<mlir::LLVM::ZExtOp>( loc, elemType, value );
                    }
                }
            }
            else
            {
                llvm_unreachable( "AssignOp lowering: expect only fixed size floating or integer types." );
            }

            rewriter.create<LLVM::StoreOp>( loc, value, allocaOp );
            rewriter.eraseOp( op );
            return success();
        }
    };

    class LoadOpLowering : public ConversionPattern
    {
       private:
        loweringContext& lState;

       public:
        LoadOpLowering( loweringContext& lState_, MLIRContext* context )
            : ConversionPattern( toy::LoadOp::getOperationName(), 1, context ), lState{ lState_ }
        {
        }

        LogicalResult matchAndRewrite( Operation* op, ArrayRef<Value> operands,
                                       ConversionPatternRewriter& rewriter ) const override
        {
            auto loadOp = cast<toy::LoadOp>( op );
            auto loc = loadOp.getLoc();

            // %0 = toy.load "i1v" : i1
            LLVM_DEBUG( llvm::dbgs() << "Lowering toy.load: " << *op << '\n' );

            auto name = loadOp.getName();
            auto allocaOp = lState.symbolToAlloca[name];

            // name: i1v
            LLVM_DEBUG( llvm::dbgs() << "name: " << name << '\n' );

            Type elemType = allocaOp.getElemType();
            LLVM_DEBUG( llvm::dbgs() << "elemType: " << elemType << '\n' );

            auto load = rewriter.create<LLVM::LoadOp>( loc, elemType, allocaOp );
            LLVM_DEBUG( llvm::dbgs() << "new load op: " << load << '\n' );

            rewriter.replaceOp( op, load.getResult() );
            return success();
        }
    };

    // Lower toy.return to nothing (erase).
    class ExitOpLowering : public ConversionPattern
    {
       private:
        loweringContext& lState;

       public:
        ExitOpLowering( loweringContext& lState_, MLIRContext* context )
            : ConversionPattern( toy::ExitOp::getOperationName(), 1, context ), lState{ lState_ }
        {
        }

        LogicalResult matchAndRewrite( Operation* op, ArrayRef<Value> operands,
                                       ConversionPatternRewriter& rewriter ) const override
        {
            LLVM_DEBUG( llvm::dbgs() << "Lowering toy.return: " << *op << '\n' );

            mlir::Location loc = op->getLoc();

            assert( op->getNumOperands() == 0 );
            if ( op->getNumOperands() == 0 )
            {
                // RETURN; or default -> return 0
                auto zero = lState.getI32zero( loc, rewriter );
                rewriter.create<LLVM::ReturnOp>( loc, zero );
            }
#if 0
            else if ( op->getNumOperands() == 1 )
            {
                toy::ExitOp returnOp = cast<toy::ExitOp>( op );

                // RETURN 3; or RETURN x;
                auto operand = returnOp.getRc()[0];

                LLVM_DEBUG( {
                    llvm::dbgs() << "Operand before type conversions:\n";
                    operand.dump();
                } );

                // Handle memref<f64> if needed (e.g., load the value)
                if ( mlir::isa<mlir::MemRefType>( operand.getType() ) )
                {
                    auto memrefType = mlir::cast<mlir::MemRefType>( operand.getType() );

                    if ( !mlir::isa<mlir::Float64Type>( memrefType.getElementType() ) )
                    {
                        returnOp->emitError( "Expected memref<f64> for return operand" );
                        return failure();
                    }

                    operand = rewriter.create<mlir::LLVM::LoadOp>( loc, rewriter.getF64Type(), operand );
                }

                if ( mlir::isa<mlir::Float64Type>( operand.getType() ) )
                {
                    operand = rewriter.create<LLVM::FPToSIOp>( loc, rewriter.getI32Type(), operand );
                }

                auto intType = mlir::cast<mlir::IntegerType>( operand.getType() );
                auto width = intType.getWidth();
                if ( width > 32 )
                {
                    operand = rewriter.create<mlir::LLVM::TruncOp>( loc, rewriter.getI32Type(), operand );
                }
                else
                {
                    // SExtOp for sign extend.
                    operand = rewriter.create<mlir::LLVM::ZExtOp>( loc, rewriter.getI32Type(), operand );
                }

                LLVM_DEBUG( {
                    llvm::dbgs() << "Final return operand:\n";
                    operand.dump();
                } );

                rewriter.create<LLVM::ReturnOp>( loc, operand );
            }
            else
            {
                llvm_unreachable( "toy.return expects 0 or 1 operands" );
            }
#endif

            rewriter.eraseOp( op );
            return success();
        }
    };

#if 0    // Now unused.
    template <class toyOpType>
    class LowerByDeletion : public ConversionPattern
    {
       public:
        LowerByDeletion( MLIRContext* context ) : ConversionPattern( toyOpType::getOperationName(), 1, context )
        {
        }

        LogicalResult matchAndRewrite( Operation* op, ArrayRef<Value> operands,
                                       ConversionPatternRewriter& rewriter ) const override
        {
            LLVM_DEBUG( llvm::dbgs() << "Lowering (by erase): " << *op << '\n' );
            rewriter.eraseOp( op );
            return success();
        }
    };
#endif

    // Lower toy.print to a call to __toy_print.
    class PrintOpLowering : public ConversionPattern
    {
       private:
        loweringContext& lState;

       public:
        PrintOpLowering( loweringContext& lState_, MLIRContext* context )
            : ConversionPattern( toy::PrintOp::getOperationName(), 1, context ), lState{ lState_ }
        {
        }

        LogicalResult matchAndRewrite( Operation* op, ArrayRef<Value> operands,
                                       ConversionPatternRewriter& rewriter ) const override
        {
            auto printOp = cast<toy::PrintOp>( op );
            auto loc = printOp.getLoc();

            LLVM_DEBUG( llvm::dbgs() << "Lowering toy.print: " << *op << '\n' );

            auto input = printOp.getInput();

            rewriter.create<LLVM::CallOp>( loc, lState.printFunc, ValueRange{ input } );

            // Erase the print op
            rewriter.eraseOp( op );
            return success();
        }
    };

    // Lower toy.negate to LLVM arithmetic.
    class NegOpLowering : public ConversionPattern
    {
       private:
        loweringContext& lState;

       public:
        NegOpLowering( loweringContext& lState_, MLIRContext* context )
            : ConversionPattern( toy::NegOp::getOperationName(), 1, context ), lState{ lState_ }
        {
        }

        LogicalResult matchAndRewrite( Operation* op, ArrayRef<Value> operands,
                                       ConversionPatternRewriter& rewriter ) const override
        {
            auto unaryOp = cast<toy::NegOp>( op );
            auto loc = unaryOp.getLoc();
            auto operand = operands[0];

            LLVM_DEBUG( llvm::dbgs() << "Lowering toy.negate: " << *op << '\n' );

            Value result = operand;

            auto zero = lState.getF64zero( loc, rewriter );
            result = rewriter.create<LLVM::FSubOp>( loc, zero, result );

            rewriter.replaceOp( op, result );
            return success();
        }
    };

    // Lower toy.binary to LLVM arithmetic.
    template <class ToyBinaryOpType, class llvmIOpType, class llvmFOpType>
    class BinaryOpLowering : public ConversionPattern
    {
       private:
        loweringContext& lState;

       public:
        BinaryOpLowering( loweringContext& lState_, MLIRContext* ctx )
            : ConversionPattern( ToyBinaryOpType::getOperationName(), 1, ctx ), lState{ lState_ }
        {
        }

        LogicalResult matchAndRewrite( Operation* op, ArrayRef<Value> operands,
                                       ConversionPatternRewriter& rewriter ) const override
        {
            auto binaryOp = cast<ToyBinaryOpType>( op );
            auto loc = binaryOp.getLoc();

            LLVM_DEBUG( llvm::dbgs() << "Lowering toy.binary: " << *op << '\n' );

            auto resultType = binaryOp.getResult().getType();

            Value lhs = operands[0];
            Value rhs = operands[1];
            if ( resultType.isIntOrIndex() )
            {
                auto result = rewriter.create<llvmIOpType>( loc, lhs, rhs );
                rewriter.replaceOp( op, result );
            }
            else
            {
                auto result = rewriter.create<llvmFOpType>( loc, lhs, rhs );
                rewriter.replaceOp( op, result );
            }

            return success();
        }
    };

    using AddOpLowering = BinaryOpLowering<toy::AddOp, LLVM::AddOp, LLVM::FAddOp>;
    using SubOpLowering = BinaryOpLowering<toy::SubOp, LLVM::SubOp, LLVM::FSubOp>;
    using MulOpLowering = BinaryOpLowering<toy::MulOp, LLVM::MulOp, LLVM::FMulOp>;
    using DivOpLowering = BinaryOpLowering<toy::DivOp, LLVM::SDivOp, LLVM::FDivOp>;

    // Lower arith.constant to LLVM constant.
    class ConstantOpLowering : public ConversionPattern
    {
       private:
        loweringContext& lState;

       public:
        ConstantOpLowering( loweringContext& lState_, MLIRContext* ctx )
            : ConversionPattern( arith::ConstantOp::getOperationName(), 1, ctx ), lState{ lState_ }
        {
        }

        LogicalResult matchAndRewrite( Operation* op, ArrayRef<Value> operands,
                                       ConversionPatternRewriter& rewriter ) const override
        {
            auto constantOp = cast<arith::ConstantOp>( op );
            auto loc = constantOp.getLoc();
            auto valueAttr = constantOp.getValue();

            LLVM_DEBUG( llvm::dbgs() << "Lowering arith.constant: " << *op << '\n' );

            if ( auto fAttr = dyn_cast<FloatAttr>( valueAttr ) )
            {
                auto f64Type = rewriter.getF64Type();    // Returns Float64Type for f64
                auto value = rewriter.create<LLVM::ConstantOp>( loc, f64Type, fAttr );
                rewriter.replaceOp( op, value );
                return success();
            }
            else if ( auto intAttr = dyn_cast<IntegerAttr>( valueAttr ) )
            {
                auto i64Type = IntegerType::get( rewriter.getContext(), 64 );
                auto value = rewriter.create<LLVM::ConstantOp>( loc, i64Type, intAttr );
                rewriter.replaceOp( op, value );
                return success();
            }

            return failure();
        }
    };

    class ToyToLLVMLoweringPass : public PassWrapper<ToyToLLVMLoweringPass, OperationPass<ModuleOp>>
    {
       public:
        MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID( ToyToLLVMLoweringPass )

        ToyToLLVMLoweringPass( toy::driverState* pst_ ) : pDriverState{ pst_ }
        {
        }

        void getDependentDialects( DialectRegistry& registry ) const override
        {
            registry.insert<LLVM::LLVMDialect, arith::ArithDialect>();
        }

        void runOnOperation() override
        {
            ModuleOp module = getOperation();

            LLVM_DEBUG( {
                llvm::dbgs() << "Starting ToyToLLVMLoweringPass on:\n";
                module->dump();
            } );

            // Create DICompileUnit
            OpBuilder builder( module.getRegion() );
            loweringContext lState;

            // Set debug metadata
            auto ctx = builder.getContext();

            // Add __toy_print declaration
            builder.setInsertionPointToStart( module.getBody() );
            auto printFuncType =
                LLVM::LLVMFunctionType::get( LLVM::LLVMVoidType::get( ctx ), { builder.getF64Type() }, false );
            lState.printFunc = builder.create<LLVM::LLVMFuncOp>( module.getLoc(), "__toy_print", printFuncType, LLVM::Linkage::External );

            // Create main function
            auto mainFuncType = LLVM::LLVMFunctionType::get( builder.getI32Type(), {}, false );
            lState.mainFunc =
                builder.create<LLVM::LLVMFuncOp>( module.getLoc(), "main", mainFuncType, LLVM::Linkage::External );

            // Initialize the type converter
            LLVMTypeConverter typeConverter( &getContext() );

            // Conversion target: only LLVM dialect is legal
            ConversionTarget target1( getContext() );
            target1.addLegalDialect<LLVM::LLVMDialect>();
            target1.addIllegalOp<arith::ConstantOp>();
            target1.addIllegalOp<toy::DeclareOp, toy::AssignOp, toy::PrintOp, toy::AddOp, toy::SubOp, toy::MulOp,
                                 toy::DivOp, toy::NegOp, toy::ExitOp>();
            target1.addLegalOp<mlir::ModuleOp>();
            target1.addIllegalDialect<toy::ToyDialect>();

            // Patterns for toy dialect and standard ops
            RewritePatternSet patterns1( &getContext() );

            // The operator ordering here doesn't matter, as there's a graph walk to find all the operator nodes,
            // and the order is based on that walk (i.e.: ProgramOpLowering happens first.)
            patterns1.insert<DeclareOpLowering>( lState, &getContext() );
            patterns1.insert<LoadOpLowering>( lState, &getContext() );
            patterns1.insert<AddOpLowering>( lState, &getContext() );
            patterns1.insert<SubOpLowering>( lState, &getContext() );
            patterns1.insert<MulOpLowering>( lState, &getContext() );
            patterns1.insert<DivOpLowering>( lState, &getContext() );
            patterns1.insert<NegOpLowering>( lState, &getContext() );
            patterns1.insert<PrintOpLowering>( lState, &getContext() );
            patterns1.insert<ConstantOpLowering>( lState, &getContext() );
            patterns1.insert<AssignOpLowering>( lState, &getContext() );
            patterns1.insert<ExitOpLowering>( lState, &getContext() );
            patterns1.insert<ProgramOpLowering>( lState, &getContext() );

            arith::populateArithToLLVMConversionPatterns( typeConverter, patterns1 );

            if ( failed( applyFullConversion( module, target1, std::move( patterns1 ) ) ) )
            {
                LLVM_DEBUG( llvm::dbgs() << "Conversion failed\n" );
                signalPassFailure();
                return;
            }

            LLVM_DEBUG( {
                llvm::dbgs() << "After ToyToLLVMLoweringPass:\n";
                for ( Operation& op : module->getRegion( 0 ).front() )
                {
                    op.dump();
                }
            } );
        }

       private:
        toy::driverState* pDriverState;
    };
}    // namespace toy

namespace mlir
{
    // Parameterless version for TableGen
    std::unique_ptr<Pass> createToyToLLVMLoweringPass()
    {
        return createToyToLLVMLoweringPass( nullptr );    // Default to no optimization
    }

    // Parameterized version
    std::unique_ptr<Pass> createToyToLLVMLoweringPass( toy::driverState* pst )
    {
        return std::make_unique<toy::ToyToLLVMLoweringPass>( pst );
    }

    // Custom registration with bool parameter
    void registerToyToLLVMLoweringPass( toy::driverState* pst )
    {
        ::mlir::registerPass( [pst]() -> std::unique_ptr<::mlir::Pass>
                              { return mlir::createToyToLLVMLoweringPass( pst ); } );
    }
}    // namespace mlir

// vim: et ts=4 sw=4
