#include "ToyDialect.h"
#include "ToyToLLVMLowering.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;

namespace {

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

    // Convert operand to f64 if necessary (e.g., from i64 constant).
    auto f64Type = FloatType::getF64(rewriter.getContext());
    Value operand = operands[0];
    if (operand.getType().isInteger(64)) {
      operand = rewriter.create<LLVM::SIToFPOp>(loc, f64Type, operand);
    }

    Value result = operand;
    if (opName == "-") {
      // Negate: fsub 0.0, value
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

    // Convert operands to f64 if necessary (e.g., from i64 constants).
    auto f64Type = FloatType::getF64(rewriter.getContext());
    Value lhs = operands[0];
    if (lhs.getType().isInteger(64)) {
      lhs = rewriter.create<LLVM::SIToFPOp>(loc, f64Type, lhs);
    }
    Value rhs = operands[1];
    if (rhs.getType().isInteger(64)) {
      rhs = rewriter.create<LLVM::SIToFPOp>(loc, f64Type, rhs);
    }

    Value result;
    if (opName == "+") {
      result = rewriter.create<LLVM::FAddOp>(loc, lhs, rhs);
    } else if (opName == "-") {
      result = rewriter.create<LLVM::FSubOp>(loc, lhs, rhs);
    } else if (opName == "*") {
      result = rewriter.create<LLVM::FMulOp>(loc, lhs, rhs);
    } else if (opName == "/") {
      result = rewriter.create<LLVM::FDivOp>(loc, lhs, rhs);
    } else {
      return failure(); // Invalid operator (should be caught by verify).
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

// Lower arith.constant to LLVM constant.
struct ConstantOpLowering : public ConversionPattern {
  ConstantOpLowering(MLIRContext *ctx)
      : ConversionPattern(arith::ConstantOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto constantOp = cast<arith::ConstantOp>(op);
    auto loc = constantOp.getLoc();
    auto valueAttr = constantOp.getValue();

    if (auto intAttr = dyn_cast<IntegerAttr>(valueAttr)) {
      auto i64Type = IntegerType::get(rewriter.getContext(), 64);
      auto value = rewriter.create<LLVM::ConstantOp>(loc, i64Type, intAttr);
      rewriter.replaceOp(op, value);
      return success();
    }

    return failure(); // Only handle i64 constants for now.
  }
};

class ToyToLLVMLoweringPass
    : public PassWrapper<ToyToLLVMLoweringPass, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, arith::ArithDialect, memref::MemRefDialect,
                    cf::ControlFlowDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Define the conversion target: only LLVM dialect is allowed.
    ConversionTarget target(getContext());
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addIllegalDialect<toy::ToyDialect, arith::ArithDialect,
                             memref::MemRefDialect>();

    // Define conversion patterns.
    RewritePatternSet patterns(&getContext());
    patterns.add<ProgramOpLowering, DeclareOpLowering, PrintOpLowering,
                 AssignOpLowering, UnaryOpLowering, BinaryOpLowering,
                 ConstantOpLowering>(&getContext());

    // Apply the conversion.
    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> createToyToLLVMLoweringPass() {
  return std::make_unique<ToyToLLVMLoweringPass>();
}
