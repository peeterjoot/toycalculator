//
// @file    calculator.md
// @author  Peeter Joot <peeterjoot@pm.me>
// @brief   This implements the Toy compiler dialect.
//
// @description
//
// This dialect implements:
// - a couple simple numeric operators (unary negation, binary +-*/),
// - an exit operation,
// - a declare operation,
// - an assignment operation, and
// - a print operation.
//
include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/SymbolInterfaces.td"

def Toy_Dialect : Dialect {
  let name = "toy";
  let summary = "A compiler MLIR dialect for a simple calculator with variables";
  let cppNamespace = "toy";
}

def Toy_ProgramOp : Op<Toy_Dialect, "program"> {
  let summary = "Program operation";
  let arguments = (ins);
  let results = (outs);
  let regions = (region AnyRegion:$body);
  let traits = [AutomaticAllocationScope];

  // Only print the region, omit types since they are always unit.
  let assemblyFormat = "$body attr-dict";
}

def Toy_DeclareOp : Op<Toy_Dialect, "declare"> {
  let summary = "Declare a variable, specifying its name and type.";
  let arguments = (ins StrAttr:$name, TypeAttr:$type);
  let results = (outs);

  // Default assemblyFormat gives:
  //   "toy.declare"() {name = "x", type = i32} : () -> ()
  // whereas with this, I'll get a more compact representation like:
  //   toy.declare "x" : i32
  //
  let assemblyFormat = "$name `:` $type attr-dict";
}

// Define a type constraint for AnyInteger or AnyFloat
def IntegerOrFloatType : TypeConstraint<"IntegerOrFloatType", CPred<[{
  return type.isa<IntegerType>() || type.isa<FloatType>();
}]>>;

def Toy_AssignOp : Op<Toy_Dialect, "assign"> {
  let summary = "Assign a value to a variable associated with a declaration";
  let arguments = (ins StrAttr:$name, IntegerOrFloatType:$value);
  let results = (outs);

  // toy.assign "x", %0 : i32
  let assemblyFormat = "$name `,` $value `:` type($value) attr-dict";
}

def Toy_LoadOp : Op<Toy_Dialect, "load"> {
  let summary = "Load a variable’s value by symbol name, returning a value of the given type.";
  let arguments = (ins StrAttr:$name);
  let results = (outs IntegerOrFloatType:$value);

  // Example: %x = toy.load "i1" : i1
  let assemblyFormat = "$name `:` type($value) attr-dict";
}

def Toy_ExitOp : Op<Toy_Dialect, "exit"> {
  let summary = "Exit from a program";
  let arguments = (ins Variadic<IntegerOrFloatType>:$rc);
  let results = (outs);
  let traits = [Terminator];

  // $rc is optional, and defaults to 0.
  let assemblyFormat = "($rc^ `:` type($rc))? attr-dict";

  let extraClassDeclaration = [{
    llvm::LogicalResult verify() {
      if (getNumOperands() > 1)
        return emitOpError("expects 0 or 1 operands");
      return llvm::success();
      }
  }];
}

def Toy_PrintOp : Op<Toy_Dialect, "print"> {
  let summary = "Print a variable";
  let arguments = (ins IntegerOrFloatType:$input);
  let results = (outs);
  let assemblyFormat = "$input `:` type($input) attr-dict";
}

def Toy_NegOp : Op<Toy_Dialect, "negate"> {
  let summary = "Negation operation (e.g., -x)";
  let arguments = (ins IntegerOrFloatType:$lhs);
  let results = (outs IntegerOrFloatType:$result);
  let assemblyFormat = "`-` $lhs `:` type($lhs) attr-dict";
}

def Toy_AddOp : Op<Toy_Dialect, "add"> {
  let summary = "Addition operation (e.g., x + y)";
  let arguments = (ins IntegerOrFloatType:$lhs, IntegerOrFloatType:$rhs);
  let results = (outs IntegerOrFloatType:$result);
  let assemblyFormat = "$lhs `+` $rhs `:` type($lhs) attr-dict";
}

def Toy_SubOp : Op<Toy_Dialect, "sub"> {
  let summary = "Subtraction operation (e.g., x - y)";
  let arguments = (ins IntegerOrFloatType:$lhs, IntegerOrFloatType:$rhs);
  let results = (outs IntegerOrFloatType:$result);
  let assemblyFormat = "$lhs `-` $rhs `:` type($lhs) attr-dict";
}

def Toy_MulOp : Op<Toy_Dialect, "mul"> {
  let summary = "Multiplication operation (e.g., x * y)";
  let arguments = (ins IntegerOrFloatType:$lhs, IntegerOrFloatType:$rhs);
  let results = (outs IntegerOrFloatType:$result);
  let assemblyFormat = "$lhs `*` $rhs `:` type($lhs) attr-dict";
}

def Toy_DivOp : Op<Toy_Dialect, "div"> {
  let summary = "Division operation (e.g., x / y)";
  let arguments = (ins IntegerOrFloatType:$lhs, IntegerOrFloatType:$rhs);
  let results = (outs IntegerOrFloatType:$result);
  let assemblyFormat = "$lhs `/` $rhs `:` type($lhs) attr-dict";
}

def Toy_LessOp : Op<Toy_Dialect, "less"> {
  let summary = "Less than operation (e.g., x < y)";
  let arguments = (ins IntegerOrFloatType:$lhs, IntegerOrFloatType:$rhs);
  let results = (outs I1:$result);
  let assemblyFormat = "$lhs `<` $rhs `:` type($lhs) attr-dict";
}

def Toy_LessEqualOp : Op<Toy_Dialect, "less_eq"> {
  let summary = "Less than or equal operation (e.g., x <= y)";
  let arguments = (ins IntegerOrFloatType:$lhs, IntegerOrFloatType:$rhs);
  let results = (outs I1:$result);
  let assemblyFormat = "$lhs `<=` $rhs `:` type($lhs) attr-dict";
}

def Toy_EqualOp : Op<Toy_Dialect, "eq"> {
  let summary = "Equal operation (e.g., x == y)";
  let arguments = (ins IntegerOrFloatType:$lhs, IntegerOrFloatType:$rhs);
  let results = (outs I1:$result);
  let assemblyFormat = "$lhs `==` $rhs `:` type($lhs) attr-dict";
}

def Toy_NotEqualOp : Op<Toy_Dialect, "neq"> {
  let summary = "Not equal operation (e.g., x != y)";
  let arguments = (ins IntegerOrFloatType:$lhs, IntegerOrFloatType:$rhs);
  let results = (outs I1:$result);
  let assemblyFormat = "$lhs `!=` $rhs `:` type($lhs) attr-dict";
}

def Toy_OrOp : Op<Toy_Dialect, "or"> {
  let summary = "Bitwise or boolean OR operation (e.g., x | y)";
  let arguments = (ins IntegerType:$lhs, IntegerType:$rhs);
  let results = (outs I1:$result);
  let assemblyFormat = "$lhs `|` $rhs `:` type($lhs) attr-dict";
}

def Toy_AndOp : Op<Toy_Dialect, "and"> {
  let summary = "Bitwise or boolean AND operation (e.g., x & y)";
  let arguments = (ins IntegerType:$lhs, IntegerType:$rhs);
  let results = (outs I1:$result);
  let assemblyFormat = "$lhs `&` $rhs `:` type($lhs) attr-dict";
}

def Toy_XorOp : Op<Toy_Dialect, "xor"> {
  let summary = "Bitwise or boolean XOR operation (e.g., x ^ y)";
  let arguments = (ins IntegerType:$lhs, IntegerType:$rhs);
  let results = (outs I1:$result);
  let assemblyFormat = "$lhs `^` $rhs `:` type($lhs) attr-dict";
}

def Toy_NotOp : Op<Toy_Dialect, "not"> {
  let summary = "Logical NOT operation (e.g., !x)";
  let arguments = (ins IntegerType:$lhs);
  let results = (outs I1:$result);
  let assemblyFormat = "`!` $lhs `:` type($lhs) attr-dict";
}
