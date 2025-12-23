//
// @file    calculator.md
// @author  Peeter Joot <peeterjoot@pm.me>
// @brief   This implements the Toy compiler dialect.
//
// @description
//
// This dialect implements:
// - A couple of simple numeric operators (unary negation, binary +-*/),
// - An exit operation,
// - A declare operation,
// - An assignment operation, and
// - A print operation.
// - A Function operation (for the implicit main, and any user defined functions.)
//
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Dialect/LLVMIR/LLVMTypes.td"

def Toy_Dialect : Dialect {
  let name = "toy";
  let summary = "A compiler MLIR dialect for a simple calculator with variables";
  let cppNamespace = "toy";
}

def Toy_DeclareOp : Op<Toy_Dialect, "declare"> {
  let summary = "Declare a parameter, variable or array, type (integer or float), and optional size.";
  let description = [{
    Declares a variable (scalar or array) or function parameter with a name and type.
    For parameters, the `parameter` attribute is set, and `param_number` indicates the parameter index.
  }];
  let arguments = (ins TypeAttr:$type, OptionalAttr<I64Attr>:$size, OptionalAttr<UnitAttr>:$parameter, OptionalAttr<I64Attr>:$param_number);
  let results = (outs);
  let traits = [Symbol];

  let extraClassDeclaration = [{
    mlir::StringAttr getSymNameAttr() { return mlir::dyn_cast<mlir::StringAttr>( getOperation()->getAttr("sym_name") ); }

    bool isParameter() { return getOperation()->hasAttr("parameter"); }

    mlir::LogicalResult verify() {
      if (!getSymNameAttr()) {
        return emitOpError("requires a 'sym_name' attribute of type StringAttr.");
      }
      if (isParameter() && !getParamNumberAttr()) {
        return emitOpError("parameter declarations require a 'param_number' attribute.");
      }
      if (!isParameter() && getParamNumberAttr()) {
        return emitOpError("non-parameter declarations cannot have a 'param_number' attribute.");
      }
      return mlir::success();
    }
  }];
}

def Toy_IntOrFloat : TypeConstraint<
  CPred<"mlir::isa<mlir::IntegerType>($_self) || mlir::isa<mlir::FloatType>($_self)">,
  "integer or float type">;

def Toy_AssignOp : Op<Toy_Dialect, "assign"> {
  let summary = "Assign a value to a variable (scalar or array element).";

  let description = [{
    Assigns `value` to the variable referenced by `var_name`.
    If `index` is present, the assignment targets the array element at that index.
    The target variable must have been declared with a matching `toy.declare`.
  }];

  let arguments = (ins
    SymbolRefAttr:$var_name,               // @t
    Optional<Index>:$index,                // optional SSA value of index type (dynamic or none)
    AnyType:$value                         // the value being assigned
  );

  let results = (outs);

  let assemblyFormat =
    "$var_name (`[` $index^ `]`)? `=` $value `:` type($value) attr-dict";
}

def Toy_LoadOp : Op<Toy_Dialect, "load"> {
  let summary = "Load a variable (scalar or array element) by symbol reference.";
  let arguments = (ins
    SymbolRefAttr:$var_name,               // @t
    Optional<Index>:$index                 // optional SSA value of index type (dynamic or none)
  );

  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$var_name (`[` $index^ `]`)? `:` type($result) attr-dict";
}

def Toy_YieldOp : Op<Toy_Dialect, "yield"> {
  let summary = "Terminator for toy.scope blocks";
  let description = [{
    A no-op terminator operation for `toy.scope` blocks to satisfy MLIR's block
    termination requirement. It carries no semantic meaning and is removed during
    lowering.
  }];
  let arguments = (ins);
  let results = (outs);
  let traits = [Terminator];
}

def Toy_ScopeOp : Op<Toy_Dialect, "scope"> {
  let summary = "A scoping construct for local variables in the Toy dialect";
  let description = "A scope operation that defines a region with a symbol table for local variables, nested within a function.";
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region AnyRegion:$body);
  let traits = [SymbolTable, AutomaticAllocationScope, OpAsmOpInterface];
  let extraClassDeclaration = [{
    mlir::LogicalResult verify() {
      if (!getBody().empty()) {
        if (getBody().getBlocks().size() != 1) {
          return emitOpError("expects exactly one block in the body region");
        }
      }
      return mlir::success();
    }
  }];
}

def Toy_CallOp : Op<Toy_Dialect, "call"> {
  let summary = "A call operation in the Toy dialect to invoke runtime functions.";
  let description = [{
    The `toy.call` operation represents a call to a runtime function, such as `__toy_print_string`.
    It is used as an intermediate operation during lowering from `toy.print` to `func.call`.
    The operation takes a callee name as a `FlatSymbolRefAttr` and a variadic list of operands.
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    Variadic<AnyType>:$operands
  );
  let results = (outs Variadic<AnyType>:$result);

  //let assemblyFormat = "$callee `(` type($operands) $operands `)` attr-dict";
}

def Toy_ReturnOp : Op<Toy_Dialect, "return"> {
  let summary = "Return operation for toy.scope blocks";
  let description = [{
    A return operation within a `toy.scope` that returns a value to the parent function.
    It is lowered to a `func.return` during `ScopeOp` lowering.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs);
  let traits = [Terminator];
}

def Toy_StringLiteralOp : Op<Toy_Dialect, "string_literal"> {
  let summary = "String literal operation";
  let arguments = (ins StrAttr:$value);
  let results = (outs AnyType:$result);
}

def Toy_PrintOp : Op<Toy_Dialect, "print"> {
  let summary = "Print a variable or string literal to standard output";
  let arguments = (ins AnyType:$input);
  let results = (outs);
  let assemblyFormat = "$input `:` type($input) attr-dict";
}

def Toy_GetOp : Op<Toy_Dialect, "get"> {
  let summary = "Read a scalar value from standard input";
  let description = [{
    Reads a single scalar value (integer or floating-point) from standard input
    The result type determines the format.
  }];

  let arguments = (ins);

  // Single result, restricted to integer or float types
  let results = (outs Toy_IntOrFloat:$value);

  // op name, with type inferred from result
  let assemblyFormat = "attr-dict `:` type(results)";
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

// vim: et ts=2 sw=2
