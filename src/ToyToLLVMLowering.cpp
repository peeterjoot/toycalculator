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
#include "ToyToLLVMLowering.h"

#define DEBUG_TYPE "toy-lowering"

using namespace mlir;

namespace
{
    // This structure was to pass state from runOnOperation to the lowering
    // class, now unused.
    //
    // I added this initially because my LLVM IR dump is not showing the normal
    // DI info that gets translated to DWARF.
    //
    // I suspect that this is because there's a new way of doing this, described
    // in:
    //    https://llvm.org/docs/RemoveDIsDebugInfo.html
    //
    // This new way is hidden by default.  Why I don't get DWARF in the end is
    // probably a different issue?
    struct loweringContext
    {
    };

#if 0
    mlir::FileLineColLoc getLocation( mlir::Location loc )
    {
        // Cast Location to FileLineColLoc
        auto fileLineLoc = mlir::dyn_cast<mlir::FileLineColLoc>( loc );
        if ( !fileLineLoc )
        {
            throw std::runtime_error(
                "Internal error: Expected only FileLineColLoc Location info." );
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
            : ConversionPattern( toy::ProgramOp::getOperationName(), 1,
                                 context ),
              lState{ lState_ }
        {
        }

        LogicalResult matchAndRewrite(
            Operation* op, ArrayRef<Value> operands,
            ConversionPatternRewriter& rewriter ) const override
        {
            auto programOp = cast<toy::ProgramOp>( op );
            auto loc = programOp.getLoc();

            LLVM_DEBUG( llvm::dbgs() << "Lowering toy.program: " << *op << '\n'
                                     << loc << '\n' );

            // Lookup the LLVM function for "main", created in runOnOperation
            auto module = op->getParentOfType<ModuleOp>();
            auto funcOp = module.lookupSymbol<LLVM::LLVMFuncOp>( "main" );

            // Create an entry block in the function
            Block* entryBlock = funcOp.addEntryBlock( rewriter );
            rewriter.setInsertionPointToStart( entryBlock );

            // Inline the toy.program's region into the function's entry block
            Region& programRegion = programOp.getRegion();
            if ( !programRegion.hasOneBlock() )
            {
                return rewriter.notifyMatchFailure(
                    programOp, "toy.program must have exactly one block" );
            }

            // Move the block's operations (e.g., toy.return) into the entry
            // block
            Block& programBlock = programRegion.front();
            rewriter.inlineRegionBefore( programRegion, entryBlock );
            rewriter.mergeBlocks( &programBlock, entryBlock,
                                  /* argValues */ {} );

            // Erase the original program op
            rewriter.eraseOp( op );

            // Recursively convert the inlined operations (e.g., toy.return)
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
            : ConversionPattern( toy::ExitOp::getOperationName(), 1,
                                 context ),
              lState{ lState_ }
        {
        }

        LogicalResult matchAndRewrite(
            Operation* op, ArrayRef<Value> operands,
            ConversionPatternRewriter& rewriter ) const override
        {
            toy::ExitOp returnOp = cast<toy::ExitOp>( op );
            LLVM_DEBUG( llvm::dbgs()
                        << "Lowering toy.return: " << *op << '\n' );

            mlir::Location loc = op->getLoc();

            if ( op->getNumOperands() == 0 )
            {
                // RETURN; or default -> return 0
                mlir::Value zero = rewriter.create<LLVM::ConstantOp>(
                    loc, rewriter.getI32Type(),
                    rewriter.getI32IntegerAttr( 0 ) );

                rewriter.create<LLVM::ReturnOp>( loc, zero );
            }
            else if ( op->getNumOperands() == 1 )
            {
                // RETURN 3; or RETURN x;
                auto operand = returnOp.getRc()[0];

                LLVM_DEBUG( {
                    llvm::dbgs() << "Operand before type conversions:\n";
                    operand.dump();
                } );

                // Handle memref<f64> if needed (e.g., load the value)
                if ( mlir::isa<mlir::MemRefType>( operand.getType() ) )
                {
                    auto memrefType =
                        mlir::cast<mlir::MemRefType>( operand.getType() );

                    if ( !mlir::isa<mlir::Float64Type>(
                             memrefType.getElementType() ) )
                    {
                        returnOp->emitError(
                            "Expected memref<f64> for return operand" );
                        return failure();
                    }

                    operand = rewriter.create<mlir::LLVM::LoadOp>(
                        loc, rewriter.getF64Type(), operand );
                }

                if ( mlir::isa<mlir::Float64Type>( operand.getType() ) )
                {
                    operand = rewriter.create<LLVM::FPToSIOp>(
                        loc, rewriter.getI32Type(), operand );
                }

                auto intType =
                    mlir::cast<mlir::IntegerType>( operand.getType() );
                if ( intType.getWidth() != 32 )
                {
                    operand = rewriter.create<mlir::LLVM::TruncOp>(
                        loc, rewriter.getI32Type(), operand );
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

            rewriter.eraseOp( op );
            return success();
        }
    };

    template <class toyOpType>
    class LowerByDeletion : public ConversionPattern
    {
       public:
        LowerByDeletion( MLIRContext* context )
            : ConversionPattern( toyOpType::getOperationName(), 1, context )
        {
        }

        LogicalResult matchAndRewrite(
            Operation* op, ArrayRef<Value> operands,
            ConversionPatternRewriter& rewriter ) const override
        {
            LLVM_DEBUG( llvm::dbgs()
                        << "Lowering (by erase): " << *op << '\n' );
            rewriter.eraseOp( op );
            return success();
        }
    };

    // Lower toy.print to a call to __toy_print.
    class PrintOpLowering : public ConversionPattern
    {
       private:
        loweringContext& lState;

       public:
        PrintOpLowering( loweringContext& lState_, MLIRContext* context )
            : ConversionPattern( toy::PrintOp::getOperationName(), 1, context ),
              lState{ lState_ }
        {
        }

        LogicalResult matchAndRewrite(
            Operation* op, ArrayRef<Value> operands,
            ConversionPatternRewriter& rewriter ) const override
        {
            auto printOp = cast<toy::PrintOp>( op );
            auto loc = printOp.getLoc();
            auto memref = operands[0];

            LLVM_DEBUG( llvm::dbgs() << "Lowering toy.print: " << *op << '\n' );

            // Handle both memref<f64> and !llvm.ptr
            Type memrefType = memref.getType();
            if ( !mlir::isa<MemRefType>( memrefType ) &&
                 !mlir::isa<LLVM::LLVMPointerType>( memrefType ) )
            {
                LLVM_DEBUG( llvm::dbgs()
                            << "Invalid memref type: " << memrefType << '\n' );
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

            // Call the print function
            rewriter.create<LLVM::CallOp>( loc, printFunc,
                                           ValueRange{ loadValue } );

            // Erase the print op
            rewriter.eraseOp( op );
            return success();
        }
    };

    using DeclareOpLowering = LowerByDeletion<toy::DeclareOp>;
    using AssignOpLowering = LowerByDeletion<toy::AssignOp>;

    // Lower toy.negate to LLVM arithmetic.
    class NegOpLowering : public ConversionPattern
    {
       private:
        loweringContext& lState;

       public:
        NegOpLowering( loweringContext& lState_, MLIRContext* context )
            : ConversionPattern( toy::NegOp::getOperationName(), 1, context ),
              lState{ lState_ }
        {
        }

        LogicalResult matchAndRewrite(
            Operation* op, ArrayRef<Value> operands,
            ConversionPatternRewriter& rewriter ) const override
        {
            auto unaryOp = cast<toy::NegOp>( op );
            auto loc = unaryOp.getLoc();
            auto operand = operands[0];

            LLVM_DEBUG( llvm::dbgs()
                        << "Lowering toy.negate: " << *op << '\n' );

            // Convert i64 to f64 if necessary
            Value result = operand;
#if 0    // dead code now.
            if ( operand.getType().isInteger( 64 ) )
            {
                result = rewriter.create<LLVM::SIToFPOp>(
                    loc, rewriter.getF64Type(), operand );
            }
#endif

            auto zero = rewriter.create<LLVM::ConstantOp>(
                loc, rewriter.getF64Type(), rewriter.getF64FloatAttr( 0.0 ) );
            result = rewriter.create<LLVM::FSubOp>( loc, zero, result );

            rewriter.replaceOp( op, result );
            return success();
        }
    };

    // Lower toy.binary to LLVM arithmetic.
    template <class ToyBinaryOpType, class llvmOpType>
    class BinaryOpLowering : public ConversionPattern
    {
       private:
        loweringContext& lState;

       public:
        BinaryOpLowering( loweringContext& lState_, MLIRContext* ctx )
            : ConversionPattern( ToyBinaryOpType::getOperationName(), 1, ctx ),
              lState{ lState_ }
        {
        }

        LogicalResult matchAndRewrite(
            Operation* op, ArrayRef<Value> operands,
            ConversionPatternRewriter& rewriter ) const override
        {
            auto binaryOp = cast<ToyBinaryOpType>( op );
            auto loc = binaryOp.getLoc();

            LLVM_DEBUG( llvm::dbgs()
                        << "Lowering toy.binary: " << *op << '\n' );

            // Convert operands to f64 if necessary
            Value lhs = operands[0];
#if 0    // dead code now.
            if ( lhs.getType().isInteger( 64 ) )
            {
                lhs = rewriter.create<LLVM::SIToFPOp>( loc, rewriter.getF64Type(), lhs );
            }
#endif
            Value rhs = operands[1];
#if 0    // dead code now.
            if ( rhs.getType().isInteger( 64 ) )
            {
                rhs = rewriter.create<LLVM::SIToFPOp>( loc, rewriter.getF64Type(), rhs );
            }
#endif

            auto result = rewriter.create<llvmOpType>( loc, lhs, rhs );
            rewriter.replaceOp( op, result );
            return success();
        }
    };

    using AddOpLowering = BinaryOpLowering<toy::AddOp, LLVM::FAddOp>;
    using SubOpLowering = BinaryOpLowering<toy::SubOp, LLVM::FSubOp>;
    using MulOpLowering = BinaryOpLowering<toy::MulOp, LLVM::FMulOp>;
    using DivOpLowering = BinaryOpLowering<toy::DivOp, LLVM::FDivOp>;

    // Lower arith.constant to LLVM constant.
    class ConstantOpLowering : public ConversionPattern
    {
       private:
        loweringContext& lState;

       public:
        ConstantOpLowering( loweringContext& lState_, MLIRContext* ctx )
            : ConversionPattern( arith::ConstantOp::getOperationName(), 1,
                                 ctx ),
              lState{ lState_ }
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
                        << "Lowering arith.constant: " << *op << '\n' );

            if ( auto fAttr = dyn_cast<FloatAttr>( valueAttr ) )
            {
                auto f64Type =
                    rewriter.getF64Type();    // Returns Float64Type for f64
                auto value =
                    rewriter.create<LLVM::ConstantOp>( loc, f64Type, fAttr );
                rewriter.replaceOp( op, value );
                return success();
            }
            else if ( auto intAttr = dyn_cast<IntegerAttr>( valueAttr ) )
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
    class MemRefAllocaOpLowering : public ConversionPattern
    {
       private:
        loweringContext& lState;

       public:
        MemRefAllocaOpLowering( loweringContext& lState_, MLIRContext* ctx )
            : ConversionPattern( memref::AllocaOp::getOperationName(), 1, ctx ),
              lState{ lState_ }
        {
        }

        LogicalResult matchAndRewrite(
            Operation* op, ArrayRef<Value> operands,
            ConversionPatternRewriter& rewriter ) const override
        {
            auto allocaOp = cast<memref::AllocaOp>( op );
            auto loc = allocaOp.getLoc();

            LLVM_DEBUG( llvm::dbgs()
                        << "Lowering memref.alloca: " << *op << '\n' );

            // Allocate memory for the memref type (f64)
            auto memRefType = mlir::cast<MemRefType>( allocaOp.getType() );
            auto elemType = memRefType.getElementType();
            auto ptrType = LLVM::LLVMPointerType::get( rewriter.getContext() );
            auto one = rewriter.create<LLVM::ConstantOp>(
                loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr( 1 ) );

            auto shape = memRefType.getShape();
            int64_t numElements = 1;    // Default for scalar MemRef
            if ( !shape.empty() )
            {
                llvm_unreachable( "Currently only expect scalar types." );

                numElements =
                    std::accumulate( shape.begin(), shape.end(), int64_t{ 1 },
                                     std::multiplies<int64_t>() );
            }

            unsigned elemSizeInBits =
                elemType.getIntOrFloatBitWidth();    // Example: For i16, this
                                                     // is 16 bits
            unsigned elemSizeInBytes = ( elemSizeInBits + 7 ) / 8;

            // Compute total size in bytes
            int64_t totalSizeInBytes = numElements * elemSizeInBytes;

            auto newAllocaOp = rewriter.create<LLVM::AllocaOp>(
                loc, ptrType, elemType, one, totalSizeInBytes );

            rewriter.replaceOp( op, newAllocaOp.getResult() );
            return success();
        }
    };

    // Lower memref.store to llvm.store.
    class MemRefStoreOpLowering : public ConversionPattern
    {
       private:
        loweringContext& lState;

       public:
        MemRefStoreOpLowering( loweringContext& lState_, MLIRContext* ctx )
            : ConversionPattern( memref::StoreOp::getOperationName(), 1, ctx ),
              lState{ lState_ }
        {
        }

        LogicalResult matchAndRewrite(
            Operation* op, ArrayRef<Value> operands,
            ConversionPatternRewriter& rewriter ) const override
        {
            auto storeOp = cast<memref::StoreOp>( op );
            auto loc = storeOp.getLoc();

            LLVM_DEBUG( llvm::dbgs()
                        << "Lowering memref.store: " << *op << '\n' );

            // Ensure the second operand is a pointer
            if ( !mlir::isa<LLVM::LLVMPointerType>( operands[1].getType() ) &&
                 !mlir::isa<MemRefType>( operands[1].getType() ) )
            {
                LLVM_DEBUG( llvm::dbgs() << "Invalid store pointer type: "
                                         << operands[1].getType() << '\n' );
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
    class MemRefLoadOpLowering : public ConversionPattern
    {
       private:
        loweringContext& lState;

       public:
        MemRefLoadOpLowering( loweringContext& lState_, MLIRContext* ctx )
            : ConversionPattern( memref::LoadOp::getOperationName(), 1, ctx ),
              lState{ lState_ }
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
                        << "Lowering memref.load: " << *op << '\n' );

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

        ToyToLLVMLoweringPass( toy::driverState* pst_ ) : pDriverState{ pst_ }
        {
        }

        void getDependentDialects( DialectRegistry& registry ) const override
        {
            registry.insert<LLVM::LLVMDialect, arith::ArithDialect,
                            memref::MemRefDialect>();
        }

        void runOnOperation() override
        {
            ModuleOp module = getOperation();

            LLVM_DEBUG( {
                llvm::dbgs() << "Starting ToyToLLVMLoweringPass on:\n";
                module->dump();
            } );

            // Initialize the type converter
            LLVMTypeConverter typeConverter( &getContext() );

            // Conversion target: only LLVM dialect is legal
            ConversionTarget target( getContext() );
            target.addLegalDialect<LLVM::LLVMDialect>();
            target.addIllegalDialect<toy::ToyDialect>();
            target.addIllegalOp<memref::AllocaOp, memref::StoreOp,
                                memref::LoadOp, arith::ConstantOp>();
            target.addLegalOp<mlir::ModuleOp>();

            // Create DICompileUnit
            OpBuilder builder( module.getRegion() );
            loweringContext lState;

            // Set debug metadata
            auto ctx = builder.getContext();

            // Add __toy_print declaration
            builder.setInsertionPointToStart( module.getBody() );
            auto printFuncType =
                LLVM::LLVMFunctionType::get( LLVM::LLVMVoidType::get( ctx ),
                                             { builder.getF64Type() }, false );
            builder.create<LLVM::LLVMFuncOp>( module.getLoc(), "__toy_print",
                                              printFuncType,
                                              LLVM::Linkage::External );

            // Create main function
            auto mainFuncType =
                LLVM::LLVMFunctionType::get( builder.getI32Type(), {}, false );
            builder.create<LLVM::LLVMFuncOp>( module.getLoc(), "main",
                                              mainFuncType,
                                              LLVM::Linkage::External );

            // Patterns for toy dialect and standard ops
            RewritePatternSet patterns( &getContext() );

            patterns.insert<ProgramOpLowering>( lState, &getContext() );
            patterns.insert<AddOpLowering>( lState, &getContext() );
            patterns.insert<SubOpLowering>( lState, &getContext() );
            patterns.insert<MulOpLowering>( lState, &getContext() );
            patterns.insert<DivOpLowering>( lState, &getContext() );
            patterns.insert<DeclareOpLowering>( &getContext() );
            patterns.insert<AssignOpLowering>( &getContext() );
            patterns.insert<NegOpLowering>( lState, &getContext() );
            patterns.insert<PrintOpLowering>( lState, &getContext() );
            patterns.insert<ConstantOpLowering>( lState, &getContext() );
            patterns.insert<MemRefAllocaOpLowering>( lState, &getContext() );
            patterns.insert<MemRefStoreOpLowering>( lState, &getContext() );
            patterns.insert<MemRefLoadOpLowering>( lState, &getContext() );
            patterns.insert<ExitOpLowering>( lState, &getContext() );

            arith::populateArithToLLVMConversionPatterns( typeConverter,
                                                          patterns );

            if ( failed( applyFullConversion( module, target,
                                              std::move( patterns ) ) ) )
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
}    // namespace


namespace mlir
{
    // Parameterless version for TableGen
    std::unique_ptr<Pass> createToyToLLVMLoweringPass()
    {
        return createToyToLLVMLoweringPass(
            nullptr );    // Default to no optimization
    }

    // Parameterized version
    std::unique_ptr<Pass> createToyToLLVMLoweringPass( toy::driverState* pst )
    {
        return std::make_unique<ToyToLLVMLoweringPass>( pst );
    }

    // Custom registration with bool parameter
    void registerToyToLLVMLoweringPass( toy::driverState* pst )
    {
        ::mlir::registerPass(
            [pst]() -> std::unique_ptr<::mlir::Pass>
            { return mlir::createToyToLLVMLoweringPass( pst ); } );
    }
}    // namespace mlir

// vim: et ts=4 sw=4
