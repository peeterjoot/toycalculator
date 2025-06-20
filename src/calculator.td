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
include "mlir/IR/BuiltinAttributes.td"

def Toy_Dialect : Dialect {
  let name = "toy";
  let summary = "A compiler MLIR dialect for a simple calculator with variables";
  let cppNamespace = "toy";
}

def Toy_FuncOp : Op<Toy_Dialect, "func"> {
  let summary = "Function operation";
  let arguments = (ins StrAttr:$name, TypeAttr:$type);
  let results = (outs);
  let regions = (region AnyRegion:$body);
  //let assemblyFormat = "$body attr-dict";

  let traits = [SymbolTable, AutomaticAllocationScope];

//  let extraClassDeclaration = [{
//    static void build(::mlir::OpBuilder &builder, ::mlir::OperationState &state) {
//      state.addRegion()->emplaceBlock();
//    }
//  }];
}

def Toy_DeclareOp : Op<Toy_Dialect, "declare"> {
  let summary = "Declare a variable or array, type (integer or float), and optional size.";
  let arguments = (ins TypeAttr:$type, OptionalAttr<I64Attr>:$size);
  let results = (outs);
  let traits = [Symbol];

  let extraClassDeclaration = [{
    mlir::StringAttr getSymNameAttr() { return mlir::dyn_cast<mlir::StringAttr>( getOperation()->getAttr("sym_name") ); }

    mlir::LogicalResult verify() {
      if (!getSymNameAttr()) {
        return emitOpError("requires a 'sym_name' attribute of type StringAttr");
      }
      return mlir::success();
    }
  }];
}

def Toy_IntOrFloat : TypeConstraint<
  CPred<"mlir::isa<mlir::IntegerType>($_self) || mlir::isa<mlir::FloatType>($_self)">,
  "integer or float type">;

def Toy_AssignOp : Op<Toy_Dialect, "assign"> {
  let summary = "Assign a value to a variable by symbol reference.";
  let arguments = (ins SymbolRefAttr:$var_name, AnyType:$value);
  let results = (outs);
//  let hasVerifier = 1;
//  let extraClassDeclaration = [{
//    mlir::SymbolRefAttr getVarName() {
//      return getVarNameAttr();
//    }
//    mlir::LogicalResult verify() {
//      if (!getVarName()) {
//        return emitOpError("requires a 'var_name' symbol reference attribute");
//      }
//      // Optionally: Verify the referenced symbol is a toy.declare
//      auto* symbol = mlir::SymbolTable::lookupNearestSymbolFrom(getOperation(), getVarName());
//      if (!symbol || !isa<toy::DeclareOp>(symbol)) {
//        return emitOpError("symbol '" << getVarName().getValue() << "' must reference a toy.declare operation");
//      }
//      return mlir::success();
//    }
//  }];
}

def Toy_LoadOp : Op<Toy_Dialect, "load"> {
  let summary = "Load a variable by symbol reference.";
  let arguments = (ins SymbolRefAttr:$var_name);
  let results = (outs AnyType:$result);
//  let hasVerifier = 1;
//  let extraClassDeclaration = [{
//    mlir::SymbolRefAttr getVarName() {
//      return getVarNameAttr();
//    }
//    mlir::LogicalResult verify() {
//      if (!getVarName()) {
//        return emitOpError("requires a 'var_name' symbol reference attribute");
//      }
//      // Optionally: Verify the referenced symbol is a toy.declare
//      auto* symbol = mlir::SymbolTable::lookupNearestSymbolFrom(getOperation(), getVarName());
//      if (!symbol || !isa<toy::DeclareOp>(symbol)) {
//        return emitOpError("symbol '" << getVarName().getValue() << "' must reference a toy.declare operation");
//      }
//      return mlir::success();
//    }
//  }];
}

def Toy_ExitOp : Op<Toy_Dialect, "exit"> {
  let summary = "Exit from a program";
  let arguments = (ins Variadic<AnyType>:$rc);
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

def Toy_StringLiteralOp : Op<Toy_Dialect, "string_literal"> {
  let summary = "String literal operation";
  let arguments = (ins StrAttr:$value);
  let results = (outs AnyType:$result);
}

def Toy_PrintOp : Op<Toy_Dialect, "print"> {
  let summary = "Print a variable or string literal";
  let arguments = (ins AnyType:$input);
  let results = (outs);
  let assemblyFormat = "$input `:` type($input) attr-dict";
}

def Toy_NegOp : Op<Toy_Dialect, "negate"> {
  let summary = "Negation operation (e.g., -x)";
  let arguments = (ins Toy_IntOrFloat:$lhs);
  let results = (outs Toy_IntOrFloat:$result);
  //let assemblyFormat = "`-` $lhs `:` type($lhs) attr-dict";
}

def Toy_AddOp : Op<Toy_Dialect, "add"> {
  let summary = "Addition operation (e.g., x + y)";
  let arguments = (ins Toy_IntOrFloat:$lhs, Toy_IntOrFloat:$rhs);
  let results = (outs Toy_IntOrFloat:$result);
  //let assemblyFormat = "$lhs `+` $rhs `:` type($result) attr-dict";
}

def Toy_SubOp : Op<Toy_Dialect, "sub"> {
  let summary = "Subtraction operation (e.g., x - y)";
  let arguments = (ins Toy_IntOrFloat:$lhs, Toy_IntOrFloat:$rhs);
  let results = (outs Toy_IntOrFloat:$result);
  //let assemblyFormat = "$lhs `-` $rhs `:` type($result) attr-dict";
}

def Toy_MulOp : Op<Toy_Dialect, "mul"> {
  let summary = "Multiplication operation (e.g., x * y)";
  let arguments = (ins Toy_IntOrFloat:$lhs, Toy_IntOrFloat:$rhs);
  let results = (outs Toy_IntOrFloat:$result);
  //let assemblyFormat = "$lhs `*` $rhs `:` type($result) attr-dict";
}

def Toy_DivOp : Op<Toy_Dialect, "div"> {
  let summary = "Division operation (e.g., x / y)";
  let arguments = (ins Toy_IntOrFloat:$lhs, Toy_IntOrFloat:$rhs);
  let results = (outs Toy_IntOrFloat:$result);
  //let assemblyFormat = "$lhs `/` $rhs `:` type($result) attr-dict";
}

def Toy_LessOp : Op<Toy_Dialect, "less"> {
  let summary = "Less than operation (e.g., x < y)";
  let arguments = (ins Toy_IntOrFloat:$lhs, Toy_IntOrFloat:$rhs);
  let results = (outs I1:$result);
  //let assemblyFormat = "$lhs `<` $rhs `:` type($result) attr-dict";
}

def Toy_LessEqualOp : Op<Toy_Dialect, "less_eq"> {
  let summary = "Less than or equal operation (e.g., x <= y)";
  let arguments = (ins Toy_IntOrFloat:$lhs, Toy_IntOrFloat:$rhs);
  let results = (outs I1:$result);
  //let assemblyFormat = "$lhs `<=` $rhs `:` type($result) attr-dict";
}

def Toy_EqualOp : Op<Toy_Dialect, "eq"> {
  let summary = "Equal operation (e.g., x EQ y)";
  let arguments = (ins Toy_IntOrFloat:$lhs, Toy_IntOrFloat:$rhs);
  let results = (outs I1:$result);
  //let assemblyFormat = "$lhs `==` $rhs `:` type($result) attr-dict";
}

def Toy_NotEqualOp : Op<Toy_Dialect, "neq"> {
  let summary = "Not equal operation (e.g., x NE y)";
  let arguments = (ins Toy_IntOrFloat:$lhs, Toy_IntOrFloat:$rhs);
  let results = (outs I1:$result);
  //let assemblyFormat = "$lhs `!=` $rhs `:` type($result) attr-dict";
}

def Toy_OrOp : Op<Toy_Dialect, "or"> {
  let summary = "Bitwise or boolean OR operation (e.g., x OR y)";
  let arguments = (ins Toy_IntOrFloat:$lhs, Toy_IntOrFloat:$rhs);
  let results = (outs Toy_IntOrFloat:$result);
  //let assemblyFormat = "$lhs `|` $rhs `:` type($result) attr-dict";
}

def Toy_AndOp : Op<Toy_Dialect, "and"> {
  let summary = "Bitwise or boolean AND operation (e.g., x AND y)";
  let arguments = (ins Toy_IntOrFloat:$lhs, Toy_IntOrFloat:$rhs);
  let results = (outs Toy_IntOrFloat:$result);
  //let assemblyFormat = "$lhs `&` $rhs `:` type($result) attr-dict";
}

def Toy_XorOp : Op<Toy_Dialect, "xor"> {
  let summary = "Bitwise or boolean XOR operation (e.g., x XOR y)";
  let arguments = (ins Toy_IntOrFloat:$lhs, Toy_IntOrFloat:$rhs);
  let results = (outs Toy_IntOrFloat:$result);
  //let assemblyFormat = "$lhs `^` $rhs `:` type($result) attr-dict";
}
