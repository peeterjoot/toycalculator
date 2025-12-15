//
// @file    ToyPasses.td
// @author  Peeter Joot <peeterjoot@pm.me>
// @brief   Dialect for lowering ToyDialect.td mlir to LLVM-IR.
//
include "mlir/Pass/PassBase.td"

def ToyToLLVMLowering : Pass<"toy-to-llvm-lowering", "mlir::ModuleOp"> {
  let summary = "Lower Toy dialect to LLVM dialect";
  let description = [{
    This pass lowers Toy dialect operations to LLVM dialect operations.
  }];
  let constructor = "mlir::createToyToLLVMLoweringPass()";
  let dependentDialects = [
    "mlir::LLVM::LLVMDialect",
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect"
  ];
}
