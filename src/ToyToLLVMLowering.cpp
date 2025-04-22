#include "ToyDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;

namespace {
class ToyToLLVMLoweringPass
    : public PassWrapper<ToyToLLVMLoweringPass, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, StandardOpsDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Define the conversion target: only LLVM dialect is allowed.
    ConversionTarget target(getContext());
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addIllegalDialect<toy::ToyDialect>();

    // Define conversion patterns.
    RewritePatternSet patterns(&getContext());
    patterns.add<ProgramOpLowering, DeclareOpLowering, PrintOpLowering,
                 AssignOpLowering, UnaryOpLowering, BinaryOpLowering>(
        &getContext());

    // Apply the conversion.
    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

// Lower toy.program to an LLVM function.
struct ProgramOpLowering : public ConversionPattern {
  ProgramOpLowering(MLIRContext *ctx)
      : ConversionPattern(toy::ProgramOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto programOp = cast<toy::ProgramOp>(op);
    auto loc = programOp.getLoc();

    // Create an LLVM function: int main() { ... }
    auto i32Type = IntegerType::get(rewriter.getContext(), 32);
    auto funcType = LLVM::LLVMFunctionType::get(i32Type, {}, false);
    auto funcOp = rewriter.create<LLVM::LLVMFuncOp>(
        loc, "main", funcType, LLVM::Linkage::External);

    // Create the entry block.
    Block *entryBlock = funcOp.addEntryBlock();
    rewriter.setInsertionPointToStart(entryBlock);

    // Convert the program body (region) to LLVM.
    Region &body = programOp.getBody();
    if (!body.empty()) {
      if (failed(rewriter.convertRegionTypes(&body, *rewriter.getTypeConverter(),
                                             entryBlock)))
        return failure();
    }

    // Return 0.
    rewriter.create<LLVM::ReturnOp>(loc, rewriter.create<LLVM::ConstantOp>(
                                              loc, i32Type, 0));

    // Replace the program op with the function.
    rewriter.replaceOp(op, funcOp.getOperation()->getResults());
    return success();
  }
};

// Lower toy.declare to llvm.alloca.
struct DeclareOpLowering : public ConversionPattern {
  DeclareOpLowering(MLIRContext *ctx)
      : ConversionPattern(toy::DeclareOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto declareOp = cast<toy::DeclareOp>(op);
    auto loc = declareOp.getLoc();

    // Allocate memory for an f64 value.
    auto f64Type = FloatType::getF64(rewriter.getContext());
    auto ptrType = LLVM::LLVMPointerType::get(f64Type);
    auto allocaOp = rewriter.create<LLVM::AllocaOp>(
        loc, ptrType, rewriter.getI64IntegerAttr(1), 0);

    // Replace the declare op (no results).
    rewriter.replaceOp(op, allocaOp.getResult());
    return success();
  }
};

// Lower toy.print to a call to __toy_print.
struct PrintOpLowering : public ConversionPattern {
  PrintOpLowering(MLIRContext *ctx)
      : ConversionPattern(toy::PrintOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto printOp = cast<toy::PrintOp>(op);
    auto loc = printOp.getLoc();

    // Load the f64 value from the pointer (from toy.declare).
    auto f64Type = FloatType::getF64(rewriter.getContext());
    auto loadOp = rewriter.create<LLVM::LoadOp>(loc, f64Type, operands[0]);

    // Declare the __toy_print function: void __toy_print(double).
    auto funcType = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(rewriter.getContext()), {f64Type}, false);
    auto funcOp = rewriter.create<LLVM::LLVMFuncOp>(
        loc, "__toy_print", funcType, LLVM::Linkage::External);

    // Call __toy_print with the loaded value.
    rewriter.create<LLVM::CallOp>(loc, funcOp, loadOp.getResult());

    // Replace the print op (no results).
    rewriter.replaceOp(op, {});
    return success();
  }
};

// Lower toy.assign to llvm.store.
struct AssignOpLowering : public ConversionPattern {
  AssignOpLowering(MLIRContext *ctx)
      : ConversionPattern(toy::AssignOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto assignOp = cast<toy::AssignOp>(op);
    auto loc = assignOp.getLoc();

    // Store the value (operands[1]) into the pointer (operands[0]).
    rewriter.create<LLVM::StoreOp>(loc, operands[1], operands[0]);

    // Replace the assign op (no results).
    rewriter.replaceOp(op, {});
    return success();
  }
};

// Lower toy.unary to LLVM arithmetic.
struct UnaryOpLowering : public ConversionPattern {
  UnaryOpLowering(MLIRContext *ctx)
      : ConversionPattern(toy::UnaryOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto unaryOp = cast<toy::UnaryOp>(op);
    auto loc = unaryOp.getLoc();
    auto opName = unaryOp.getOp();

    Value result = operands[0];
    if (opName == "-") {
      // Negate: fsub 0.0, value
      auto f64Type = FloatType::getF64(rewriter.getContext());
      auto zero = rewriter.create<LLVM::ConstantOp>(
          loc, f64Type, rewriter.getF64FloatAttr(0.0));
      result = rewriter.create<LLVM::FSubOp>(loc, zero, result);
    }
    // "+" is a no-op (identity).

    rewriter.replaceOp(op, result);
    return success();
  }
};

// Lower toy.binary to LLVM arithmetic.
struct BinaryOpLowering : public ConversionPattern {
  BinaryOpLowering(MLIRContext *ctx)
      : ConversionPattern(toy::BinaryOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto binaryOp = cast<toy::BinaryOp>(op);
    auto loc = binaryOp.getLoc();
    auto opName = binaryOp.getOp();

    Value result;
    if (opName == "+") {
      result = rewriter.create<LLVM::FAddOp>(loc, operands[0], operands[1]);
    } else if (opName == "-") {
      result = rewriter.create<LLVM::FSubOp>(loc, operands[0], operands[1]);
    } else if (opName == "*") {
      result = rewriter.create<LLVM::FMulOp>(loc, operands[0], operands[1]);
    } else if (opName == "/") {
      result = rewriter.create<LLVM::FDivOp>(loc, operands[0], operands[1]);
    } else {
      return failure(); // Invalid operator (should be caught by verify).
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

std::unique_ptr<Pass> createToyToLLVMLoweringPass() {
  return std::make_unique<ToyToLLVMLoweringPass>();
}
