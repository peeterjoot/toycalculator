include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinTypes.td"
//include "mlir/Interfaces/InferTypeOpInterface.td"
//include "mlir/Interfaces/SideEffectInterfaces.td" // Pure

// Define the Toy dialect
def Toy_Dialect : Dialect {
  let name = "toy";
  let summary = "Toy dialect for a simple calculator";
  let cppNamespace = "toy";
}

// Program operation
def Toy_ProgramOp : Op<Toy_Dialect, "program"> {
  let summary = "Program operation";
  let arguments = (ins);
  let results = (outs);
  let regions = (region AnyRegion:$body);
  //let traits = [Pure, NoTerminator];
}

// Declare operation
def Toy_DeclareOp : Op<Toy_Dialect, "declare"> {
  let summary = "Declare a variable";
  let arguments = (ins StrAttr:$name);
  let results = (outs);
  //let traits = [Pure];
}

// Print operation
def Toy_PrintOp : Op<Toy_Dialect, "print"> {
  let summary = "Print a variable";
  //let arguments = (ins MemRefType:$input);
  let arguments = (ins AnyType:$input);
  let results = (outs);
  // No traits, as printing has side effects
}

// Assign operation
def Toy_AssignOp : Op<Toy_Dialect, "assign"> {
  let summary = "Assign a value";
  let arguments = (ins StrAttr:$name, F64:$value);
  let results = (outs);
  //let traits = [Pure];
}

// Unary operation
def Toy_UnaryOp : Op<Toy_Dialect, "unary"> {
  let summary = "Unary operation (e.g., +x, -x)";
  let arguments = (ins
    StrAttr:$op,              // Operator name (e.g., "+", "-")
    F64:$lhs              // Input operand (f64)
  );
  let results = (outs
    F64:$result           // Output (f64)
  );
  //let traits = [
  //  Pure,                     // No side effects
  //  SameOperandsAndResultType // Operands and result have the same type (f64)
  //];
  let extraClassDeclaration = [{
    // Optional: Add custom verification for valid operators
    llvm::LogicalResult verify() {
      llvm::StringRef op = getOp();
      if (op != "+" && op != "-")
        return emitOpError("invalid unary operator: ") << op;
      return llvm::success();
    }
  }];
}

// Binary operation
def Toy_BinaryOp : Op<Toy_Dialect, "binary"> {
  let summary = "Binary operation (e.g., x + y, x * y)";
  let arguments = (ins
    StrAttr:$op,              // Operator name (e.g., "+", "-", "*", "/")
    F64:$lhs,             // Left operand (f64)
    F64:$rhs              // Right operand (f64)
  );
  let results = (outs
    F64:$result           // Output (f64)
  );
  //let traits = [
  //  Pure,                     // No side effects
  //  SameOperandsAndResultType // Operands and result have the same type (f64)
  //];
  let extraClassDeclaration = [{
    // Optional: Add custom verification for valid operators
    llvm::LogicalResult verify() {
      llvm::StringRef op = getOp();
      if (op != "+" && op != "-" && op != "*" && op != "/")
        return emitOpError("invalid binary operator: ") << op;
      return llvm::success();
    }
  }];
}
