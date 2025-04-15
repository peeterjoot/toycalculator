include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

def Toy_Dialect : Dialect {
  let name = "toy";
  let summary = "Toy dialect for a simple calculator";
  let cppNamespace = "::toy";
}

def Toy_ProgramOp : Op<Toy_Dialect, "program"> {
  let summary = "Program operation";
  let arguments = (ins);
  let results = (outs);
  let regions = (region AnyRegion:$body);
}

def Toy_DeclareOp : Op<Toy_Dialect, "declare"> {
  let summary = "Declare a variable";
  let arguments = (ins StrAttr:$name);
  let results = (outs);
}

def Toy_PrintOp : Op<Toy_Dialect, "print"> {
  let summary = "Print a variable";
  let arguments = (ins StrAttr:$name);
  let results = (outs);
}

def Toy_AssignOp : Op<Toy_Dialect, "assign"> {
  let summary = "Assign a value";
  let arguments = (ins StrAttr:$name, AnyType:$value);
  let results = (outs);
}

def Toy_BinaryOp : Op<Toy_Dialect, "binary"> {
  let summary = "Binary operation";
  let arguments = (ins StrAttr:$op, AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
}
