include "mlir/Pass/PassBase.td"

def ToyToLLVMLowering : Pass<"toy-to-llvm-lowering", "mlir::ModuleOp"> {
  let summary = "Lower Toy dialect to LLVM dialect";
  let constructor = "mlir::createToyToLLVMLoweringPass()";
  let dependentDialects = [
    "mlir::LLVM::LLVMDialect",
    "mlir::StandardOpsDialect"
  ];
}
