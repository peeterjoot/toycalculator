//
// @file    SillyPasses.td
// @author  Peeter Joot <peeterjoot@pm.me>
// @brief   Dialect for lowering SillyDialect.td mlir to LLVM-IR.
//
include "mlir/Pass/PassBase.td"

def SillyToLLVMLowering : Pass<"silly-to-llvm-lowering", "mlir::ModuleOp"> {
  let summary = "Lower Silly dialect to LLVM dialect";
  let description = [{
    This pass lowers Silly dialect operations to LLVM dialect operations.
  }];
  let constructor = "mlir::createSillyToLLVMLoweringPass()";
  let dependentDialects = [
    "mlir::LLVM::LLVMDialect",
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect"
  ];
}
