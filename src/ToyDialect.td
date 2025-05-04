include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinTypes.td"

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
  //let traits = [AutomaticAllocationScope]; 
}

// Declare operation
def Toy_DeclareOp : Op<Toy_Dialect, "declare"> {
  let summary = "Declare a variable";
  let arguments = (ins StrAttr:$name);
  let results = (outs);
  //let traits = [Pure];
}

// Return operation
def Toy_ReturnOp : Op<Toy_Dialect, "return"> {
  let summary = "Return from a program";
  let arguments = (ins Variadic<AnyType>:$rc);
  let results = (outs);
  let traits = [Terminator];

  // Letting $rc be optional to match the grammar (may not add builder support for values yet.)
  let assemblyFormat = "($rc^ `:` type($rc))? attr-dict";

  let extraClassDeclaration = [{
    llvm::LogicalResult verify() {
      if (getNumOperands() > 1)
        return emitOpError("expects 0 or 1 operands");
      return llvm::success();
      }
  }];
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

def Toy_NegOp : Op<Toy_Dialect, "negate"> {
  let summary = "Negation operation (e.g., -x)";
  let arguments = (ins
    AnyType:$lhs            // Input operand (f64, i64)
//    F64:$lhs              // Input operand (f64)
  );
  let results = (outs
    F64:$result           // Output (f64)
  );
  //let traits = [
  //  Pure                      // No side effects
  //];
}

// let traits = [Pure, SameOperandsAndResultType];
def Toy_AddOp : Op<Toy_Dialect, "add"> {
  let summary = "Addition operation (e.g., x + y)";
  //let arguments = (ins F64:$lhs, F64:$rhs);
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs F64:$result);
}

def Toy_SubOp : Op<Toy_Dialect, "sub"> {
  let summary = "Subtraction operation (e.g., x - y)";
  //let arguments = (ins F64:$lhs, F64:$rhs);
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs F64:$result);
}

def Toy_MulOp : Op<Toy_Dialect, "mul"> {
  let summary = "Multiplication operation (e.g., x * y)";
  //let arguments = (ins F64:$lhs, F64:$rhs);
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs F64:$result);
}

def Toy_DivOp : Op<Toy_Dialect, "div"> {
  let summary = "Division operation (e.g., x / y)";
  //let arguments = (ins F64:$lhs, F64:$rhs);
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs F64:$result);
}
