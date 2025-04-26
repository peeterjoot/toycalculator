#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Debug.h>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Block.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

#include "ToyDialect.h"
#include "ToyToLLVMLowering.h"

#define DEBUG_TYPE "toy-lowering"

using namespace mlir;

namespace
{
    // Lower toy.program to an LLVM function.
    class ProgramOpLowering : public ConversionPattern
    {
       public:
        ProgramOpLowering( MLIRContext* context )
            : ConversionPattern( toy::ProgramOp::getOperationName(), 1,
                                 context )
        {
        }

        LogicalResult matchAndRewrite(
            Operation* op, ArrayRef<Value> operands,
            ConversionPatternRewriter& rewriter ) const override
        {
            auto programOp = cast<toy::ProgramOp>( op );
            auto loc = programOp.getLoc();

            LLVM_DEBUG( llvm::dbgs()
                        << "Lowering toy.program: " << *op << "\n" );

            // Create an LLVM function
            auto funcType =
                LLVM::LLVMFunctionType::get( rewriter.getI32Type(), {} );
            auto funcOp = rewriter.create<LLVM::LLVMFuncOp>(
                loc, "__toy_main", funcType, LLVM::Linkage::External );

            // Ensure the function has an entry block
            Block* entryBlock = funcOp.addEntryBlock( rewriter );

            // Inline the program's region into the entry block
            Region& programRegion = programOp.getBody();
            if ( !programRegion.empty() )
            {
                rewriter.inlineRegionBefore( programRegion, entryBlock );
            }
            else
            {
                LLVM_DEBUG( llvm::dbgs() << "Program region is empty\n" );
            }

            // Add a return operation at the end of the entry block
            rewriter.setInsertionPointToEnd( entryBlock );
            auto zero = rewriter.create<LLVM::ConstantOp>(
                loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr( 0 ) );
            rewriter.create<LLVM::ReturnOp>( loc, zero );

            // Erase the original program op
            rewriter.eraseOp( op );
            return success();
        }
    };

    // Lower toy.declare to nothing (erase).
    class DeclareOpLowering : public ConversionPattern
    {
       public:
        DeclareOpLowering( MLIRContext* context )
            : ConversionPattern( toy::DeclareOp::getOperationName(), 1,
                                 context )
        {
        }

        LogicalResult matchAndRewrite(
            Operation* op, ArrayRef<Value> operands,
            ConversionPatternRewriter& rewriter ) const override
        {
            LLVM_DEBUG( llvm::dbgs()
                        << "Lowering toy.declare: " << *op << "\n" );
            rewriter.eraseOp( op );
            return success();
        }
    };

    // Lower toy.print to a call to __toy_print.
    class PrintOpLowering : public ConversionPattern
    {
       public:
        PrintOpLowering( MLIRContext* context )
            : ConversionPattern( toy::PrintOp::getOperationName(), 1, context )
        {
        }

        LogicalResult matchAndRewrite(
            Operation* op, ArrayRef<Value> operands,
            ConversionPatternRewriter& rewriter ) const override
        {
            auto printOp = cast<toy::PrintOp>( op );
            auto loc = printOp.getLoc();
            auto memref = operands[0];

            LLVM_DEBUG( llvm::dbgs() << "Lowering toy.print: " << *op << "\n" );

            // Handle both memref<f64> and !llvm.ptr
            Type memrefType = memref.getType();
            if ( !mlir::isa<MemRefType>( memrefType ) &&
                 !mlir::isa<LLVM::LLVMPointerType>( memrefType ) )
            {
                LLVM_DEBUG( llvm::dbgs()
                            << "Invalid memref type: " << memrefType << "\n" );
                return failure();
            }

            // Load the value
            Value loadValue;
            if ( mlir::isa<LLVM::LLVMPointerType>( memrefType ) )
            {
                loadValue = rewriter.create<LLVM::LoadOp>(
                    loc, rewriter.getF64Type(), memref );
            }
            else
            {
                loadValue = rewriter.create<memref::LoadOp>( loc, memref,
                                                             ValueRange{} );
            }

            // Ensure print function exists
            auto module = op->getParentOfType<ModuleOp>();
            auto printFunc =
                module.lookupSymbol<LLVM::LLVMFuncOp>( "__toy_print" );
            if ( !printFunc )
            {
                auto printFuncType = LLVM::LLVMFunctionType::get(
                    LLVM::LLVMVoidType::get( rewriter.getContext() ),
                    { rewriter.getF64Type() } );
                printFunc = rewriter.create<LLVM::LLVMFuncOp>(
                    loc, "__toy_print", printFuncType,
                    LLVM::Linkage::External );
                printFunc.setPrivate();
            }

            // Call the print function
            rewriter.create<LLVM::CallOp>( loc, printFunc,
                                           ValueRange{ loadValue } );

            // Erase the print op
            rewriter.eraseOp( op );
            return success();
        }
    };

    // Lower toy.assign to nothing (erase).
    class AssignOpLowering : public ConversionPattern
    {
       public:
        AssignOpLowering( MLIRContext* context )
            : ConversionPattern( toy::AssignOp::getOperationName(), 1, context )
        {
        }

        LogicalResult matchAndRewrite(
            Operation* op, ArrayRef<Value> operands,
            ConversionPatternRewriter& rewriter ) const override
        {
            LLVM_DEBUG( llvm::dbgs()
                        << "Lowering toy.assign: " << *op << "\n" );
            rewriter.eraseOp( op );
            return success();
        }
    };

    // Lower toy.unary to LLVM arithmetic.
    class UnaryOpLowering : public ConversionPattern
    {
       public:
        UnaryOpLowering( MLIRContext* context )
            : ConversionPattern( toy::UnaryOp::getOperationName(), 1, context )
        {
        }

        LogicalResult matchAndRewrite(
            Operation* op, ArrayRef<Value> operands,
            ConversionPatternRewriter& rewriter ) const override
        {
            auto unaryOp = cast<toy::UnaryOp>( op );
            auto loc = unaryOp.getLoc();
            auto operand = operands[0];

            LLVM_DEBUG( llvm::dbgs() << "Lowering toy.unary: " << *op << "\n" );

            // Convert i64 to f64 if necessary
            Value result = operand;
            if ( operand.getType().isInteger( 64 ) )
            {
                result = rewriter.create<LLVM::SIToFPOp>(
                    loc, rewriter.getF64Type(), operand );
            }

            // Apply unary operation
            if ( unaryOp.getOp() == "-" )
            {
                auto zero = rewriter.create<LLVM::ConstantOp>(
                    loc, rewriter.getF64Type(),
                    rewriter.getF64FloatAttr( 0.0 ) );
                result = rewriter.create<LLVM::FSubOp>( loc, zero, result );
            }
            else if ( unaryOp.getOp() != "+" )
            {
                LLVM_DEBUG( llvm::dbgs() << "Unsupported unary op: "
                                         << unaryOp.getOp() << "\n" );
                return failure();
            }

            rewriter.replaceOp( op, result );
            return success();
        }
    };

    // Lower toy.binary to LLVM arithmetic.
    struct BinaryOpLowering : public ConversionPattern
    {
        BinaryOpLowering( MLIRContext* ctx )
            : ConversionPattern( toy::BinaryOp::getOperationName(), 1, ctx )
        {
        }

        LogicalResult matchAndRewrite(
            Operation* op, ArrayRef<Value> operands,
            ConversionPatternRewriter& rewriter ) const override
        {
            auto binaryOp = cast<toy::BinaryOp>( op );
            auto loc = binaryOp.getLoc();
            auto opName = binaryOp.getOp();

            LLVM_DEBUG( llvm::dbgs()
                        << "Lowering toy.binary: " << *op << "\n" );

            // Convert operands to f64 if necessary
            auto f64Type = rewriter.getF64Type();
            Value lhs = operands[0];
            if ( lhs.getType().isInteger( 64 ) )
            {
                lhs = rewriter.create<LLVM::SIToFPOp>( loc, f64Type, lhs );
            }
            Value rhs = operands[1];
            if ( rhs.getType().isInteger( 64 ) )
            {
                rhs = rewriter.create<LLVM::SIToFPOp>( loc, f64Type, rhs );
            }

            Value result;
            if ( opName == "+" )
            {
                result = rewriter.create<LLVM::FAddOp>( loc, lhs, rhs );
            }
            else if ( opName == "-" )
            {
                result = rewriter.create<LLVM::FSubOp>( loc, lhs, rhs );
            }
            else if ( opName == "*" )
            {
                result = rewriter.create<LLVM::FMulOp>( loc, lhs, rhs );
            }
            else if ( opName == "/" )
            {
                result = rewriter.create<LLVM::FDivOp>( loc, lhs, rhs );
            }
            else
            {
                return failure();
            }

            return failure();    // Only handle i64 constants for now
        }
    };

    // Lower arith.constant to LLVM constant.
    struct ConstantOpLowering : public ConversionPattern
    {
        ConstantOpLowering( MLIRContext* ctx )
            : ConversionPattern( arith::ConstantOp::getOperationName(), 1, ctx )
        {
        }

        LogicalResult matchAndRewrite(
            Operation* op, ArrayRef<Value> operands,
            ConversionPatternRewriter& rewriter ) const override
        {
            auto constantOp = cast<arith::ConstantOp>( op );
            auto loc = constantOp.getLoc();
            auto valueAttr = constantOp.getValue();

            LLVM_DEBUG( llvm::dbgs()
                        << "Lowering arith.constant: " << *op << "\n" );

            if ( auto intAttr = dyn_cast<IntegerAttr>( valueAttr ) )
            {
                auto i64Type = IntegerType::get( rewriter.getContext(), 64 );
                auto value =
                    rewriter.create<LLVM::ConstantOp>( loc, i64Type, intAttr );
                rewriter.replaceOp( op, value );
                return success();
            }

            return failure();
        }
    };

    // Lower memref.alloca to llvm.alloca.
    struct MemRefAllocaOpLowering : public ConversionPattern
    {
        MemRefAllocaOpLowering( MLIRContext* ctx )
            : ConversionPattern( memref::AllocaOp::getOperationName(), 1, ctx )
        {
        }

        LogicalResult matchAndRewrite(
            Operation* op, ArrayRef<Value> operands,
            ConversionPatternRewriter& rewriter ) const override
        {
            auto allocaOp = cast<memref::AllocaOp>( op );
            auto loc = allocaOp.getLoc();

            LLVM_DEBUG( llvm::dbgs()
                        << "Lowering memref.alloca: " << *op << "\n" );

            // Allocate memory for the memref type (f64)
            auto memRefType = mlir::cast<MemRefType>( allocaOp.getType() );
            auto elemType = memRefType.getElementType();
            auto ptrType = LLVM::LLVMPointerType::get( rewriter.getContext() );
            auto one = rewriter.create<LLVM::ConstantOp>(
                loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr( 1 ) );
            auto newAllocaOp = rewriter.create<LLVM::AllocaOp>(
                loc, ptrType, elemType, one, /*alignment=*/8 );

            rewriter.replaceOp( op, newAllocaOp.getResult() );
            return success();
        }
    };

    // Lower memref.store to llvm.store.
    struct MemRefStoreOpLowering : public ConversionPattern
    {
        MemRefStoreOpLowering( MLIRContext* ctx )
            : ConversionPattern( memref::StoreOp::getOperationName(), 1, ctx )
        {
        }

        LogicalResult matchAndRewrite(
            Operation* op, ArrayRef<Value> operands,
            ConversionPatternRewriter& rewriter ) const override
        {
            auto storeOp = cast<memref::StoreOp>( op );
            auto loc = storeOp.getLoc();

            LLVM_DEBUG( llvm::dbgs()
                        << "Lowering memref.store: " << *op << "\n" );

            // Ensure the second operand is a pointer
            if ( !mlir::isa<LLVM::LLVMPointerType>( operands[1].getType() ) &&
                 !mlir::isa<MemRefType>( operands[1].getType() ) )
            {
                LLVM_DEBUG( llvm::dbgs() << "Invalid store pointer type: "
                                         << operands[1].getType() << "\n" );
                return failure();
            }

            Value ptr = operands[1];
            if ( mlir::isa<MemRefType>( ptr.getType() ) )
            {
                // If still a memref, assume it will be lowered to a pointer
                ptr = rewriter.create<memref::LoadOp>( loc, ptr, ValueRange{} )
                          .getResult();
            }

            rewriter.create<LLVM::StoreOp>( loc, operands[0], ptr );
            rewriter.eraseOp( op );
            return success();
        }
    };

    // Lower memref.load to llvm.load.
    struct MemRefLoadOpLowering : public ConversionPattern
    {
        MemRefLoadOpLowering( MLIRContext* ctx )
            : ConversionPattern( memref::LoadOp::getOperationName(), 1, ctx )
        {
        }

        LogicalResult matchAndRewrite(
            Operation* op, ArrayRef<Value> operands,
            ConversionPatternRewriter& rewriter ) const override
        {
            auto loadOp = cast<memref::LoadOp>( op );
            auto loc = loadOp.getLoc();
            auto memRefType =
                mlir::cast<MemRefType>( loadOp.getMemRef().getType() );

            LLVM_DEBUG( llvm::dbgs()
                        << "Lowering memref.load: " << *op << "\n" );

            auto newLoadOp = rewriter.create<LLVM::LoadOp>(
                loc, memRefType.getElementType(), operands[0] );
            rewriter.replaceOp( op, newLoadOp.getResult() );
            return success();
        }
    };

    class ToyToLLVMLoweringPass
        : public PassWrapper<ToyToLLVMLoweringPass, OperationPass<ModuleOp>>
    {
       public:
        MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID( ToyToLLVMLoweringPass )

        void getDependentDialects( DialectRegistry& registry ) const override
        {
            registry.insert<LLVM::LLVMDialect, arith::ArithDialect,
                            memref::MemRefDialect>();
        }

        void runOnOperation() override
        {
            LLVM_DEBUG( {
                llvm::dbgs() << "Starting ToyToLLVMLoweringPass on:\n";
                getOperation()->dump();
            } );

            // Initialize the type converter
            LLVMTypeConverter typeConverter( &getContext() );

            ConversionTarget target( getContext() );
            target.addLegalDialect<LLVM::LLVMDialect>();
            target.addLegalOp<ModuleOp>();
            target.addIllegalDialect<toy::ToyDialect>();
            target.addIllegalOp<memref::AllocaOp, memref::StoreOp,
                                memref::LoadOp, arith::ConstantOp>();

            RewritePatternSet patterns( &getContext() );
            patterns.add<ProgramOpLowering, DeclareOpLowering, AssignOpLowering,
                         UnaryOpLowering, PrintOpLowering, ConstantOpLowering,
                         MemRefAllocaOpLowering, MemRefStoreOpLowering,
                         MemRefLoadOpLowering>( &getContext() );
            arith::populateArithToLLVMConversionPatterns( typeConverter,
                                                          patterns );

            LLVM_DEBUG( llvm::dbgs() << "Applying full conversion\n" );
            if ( failed( applyFullConversion( getOperation(), target,
                                              std::move( patterns ) ) ) )
            {
                LLVM_DEBUG( llvm::dbgs() << "Conversion failed\n" );
                signalPassFailure();
            }

            LLVM_DEBUG( {
                llvm::dbgs() << "After ToyToLLVMLoweringPass:\n";
                getOperation()->dump();
            } );
        }
    };

}    // namespace

namespace mlir
{
    std::unique_ptr<Pass> createToyToLLVMLoweringPass()
    {
        return std::make_unique<ToyToLLVMLoweringPass>();
    }
}    // namespace mlir

// vim: et ts=4 sw=4
