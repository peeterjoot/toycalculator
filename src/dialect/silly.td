///
/// @file    silly.md
/// @author  Peeter Joot <peeterjoot@pm.me>
/// @brief   This implements the Silly compiler dialect.
///
/// @description
///
/// This dialect implements:
/// - Numeric operators (unary negation, binary +-*/),
/// - An exit operation,
/// - A declare operation,
/// - An assignment operation,
/// - A print operation,
/// - enums and types associated with the above
/// - ...
///
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Dialect/LLVMIR/LLVMTypes.td"
include "mlir/IR/EnumAttr.td"

def Silly_Dialect : Dialect {
  let name = "silly";
  let summary = "An MLIR dialect for a silly simple language";
  let cppNamespace = "silly";

  let useDefaultTypePrinterParser = 1;
}

def Silly_VarType : TypeDef<Silly_Dialect, "var"> {
  let summary = "Abstract variable location (scalar or array)";
  let description = [{
    Represents an abstract handle to a variable's storage.
    Scalars have empty shape; arrays have a non-empty shape.
    This type is storage-agnostic and lowered to concrete memory
    (e.g., LLVM alloca).

    Multi-dimensional shapes are not currently supported (varType::parse)
  }];

  let parameters = (ins
    "mlir::Type":$elementType,
    "mlir::DenseI64ArrayAttr":$shape // empty = scalar
  );

  let mnemonic = "var";

  let hasCustomAssemblyFormat = 1;
}

def Silly_DeclareOp : Op<Silly_Dialect, "declare"> {
  let summary = "Declare an abstract variable location (scalar or array).";
  let description = [{
    Declares a variable (scalar or array) and it's type.
    The result is an abstract handle (!silly.var) to the variable's location, storage-agnostic.
    Initializers are provided as operands; for arrays, excess initializers are an error.
  }];
  let arguments = (ins
    Variadic<AnyType>:$initializers
  );
  let results = (outs Silly_VarType:$var);

  let hasVerifier = 1;
}

def Silly_IntOrFloat : TypeConstraint<
  CPred<"mlir::isa<mlir::IntegerType>($_self) || mlir::isa<mlir::FloatType>($_self)">,
  "integer or float type">;

def Silly_AssignOp : Op<Silly_Dialect, "assign"> {
  let summary = "Assign a value to a variable (or array element).";

  let description = [{
    Assigns `value` to the storage referenced by `%var`.
    If `index` is present, it targets the array element at that (dynamic) index.
    The `%var` must be a result of a `silly.declare` (or equivalent) with matching element type and shape.
    For scalars: `index` must be absent.
    For arrays: `index` is required (even if statically known to be 0 for single-element arrays).
  }];

  let arguments = (ins
    Silly_VarType:$var,                    // the SSA handle from declare
    Optional<Index>:$index,                // dynamic index for array element (absent = scalar)
    AnyType:$value                         // value to assign (must match elementType of $var)
  );

  let results = (outs);   // side-effecting op, no result

  let assemblyFormat = [{
    $var `:` type($var) (`[` $index^ `]`)? `=` $value `:` type($value) attr-dict
  }];
}

def Silly_DebugNameOp : Op<Silly_Dialect, "debug_name", [NoMemoryEffect]> {
  let summary = "Associates a source-level name with a value for debug information purposes.";

  let description = [{
    This operation has no runtime effect and exists solely to propagate a variable name
    (and optionally a source location) to lowering passes that emit debug metadata (e.g. LLVM IR).
    It is typically inserted immediately after the value is defined (or at the point where
    the value becomes "live" in source terms).

    Example:
      %ind = arith.addi %prev, %step : i32
      "silly.debug_name"(%ind) {name = "i"} : (i32) -> ()
  }];

  let arguments = (ins
    AnyType:$value,
    StrAttr:$name
  );
}

def Silly_LoadOp : Op<Silly_Dialect, "load"> {
  let summary = "Load a variable (scalar or array element) by symbol reference.";
  let arguments = (ins
    Silly_VarType:$var,                    // the SSA handle from declare
    Optional<Index>:$index                 // optional SSA value of index type (dynamic or none)
  );

  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$var `:` type($var) (`[` $index^ `]`)? `:` type($result) attr-dict";
}

def Silly_CallOp : Op<Silly_Dialect, "call"> {
  let summary = "A call operation in the Silly dialect to invoke runtime functions.";
  let description = [{
    The `silly.call` operation represents a call to a runtime function, such as `__silly_print_string`.
    It is used as an intermediate operation during lowering from `silly.print` to `func.call`.
    The operation takes a callee name as a `FlatSymbolRefAttr` and a variadic list of operands.
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    Variadic<AnyType>:$operands
  );

  let results = (outs Variadic<AnyType>:$result);

  //let assemblyFormat = "$callee `(` type($operands) $operands `)` attr-dict";
}

def Silly_StringLiteralOp : Op<Silly_Dialect, "string_literal"> {
  let summary = "String literal operation";
  let arguments = (ins StrAttr:$value);
  let results = (outs AnyType:$result);
}

def Silly_AbortOp : Op<Silly_Dialect, "abort"> {
  let summary = "Print a fatal error message to standard error, with file:line, and then abort";
  let arguments = (ins);
  let results = (outs);
}

def Silly_PrintOp : Op<Silly_Dialect, "print"> {
  let summary = "Print a list of values or string literals to standard output";

  // Variadic operand segment (0 or more arguments).
  let arguments = (ins I32:$flags, Variadic<AnyTypeOf<[AnyInteger, AnyFloat, LLVMPointerType]>>:$inputs);

  let results = (outs);

  // My old custom print assembly format for a single value (also before the flags were introduced.)
  //let assemblyFormat = "$input `:` type($input) attr-dict";
}

def Silly_GetOp : Op<Silly_Dialect, "get"> {
  let summary = "Read a scalar value from standard input";
  let description = [{
    Reads a single scalar value (integer or floating-point) from standard input
    The result type determines the format.
  }];

  let arguments = (ins);

  // Single result, restricted to integer or float types
  let results = (outs Silly_IntOrFloat:$value);

  // op name, with type inferred from result
  let assemblyFormat = "attr-dict `:` type(results)";
}

def Silly_ArithBinOpKind : I32EnumAttr<
    "ArithBinOpKind",
    "arithmetic binary operation kind",
    [
      I32EnumAttrCase<"Mul", 0, "mul">,
      I32EnumAttrCase<"Div", 1, "div">,
      I32EnumAttrCase<"Mod", 2, "mod">,
      I32EnumAttrCase<"Add", 3, "add">,
      I32EnumAttrCase<"Sub", 4, "sub">,
      I32EnumAttrCase<"Or",  5, "or">,
      I32EnumAttrCase<"And", 6, "and">,
      I32EnumAttrCase<"Xor", 7, "xor">
]> {
  let cppNamespace = "::silly";
}

def Silly_ArithBinOp : Op<Silly_Dialect, "arith_binop"> {
  let summary = "Generic arithmetic binary operation (mul, div, mod, etc.)";
  let description = [{
    Performs a binary arithmetic operation selected by the kind attribute.
    Both operands must have the same type. The result type matches the operands.
    Division and modulo behavior follows language semantics (integer/floating-point).
  }];

  let arguments = (ins
    Silly_ArithBinOpKind:$kind,
    Silly_IntOrFloat:$lhs,
    Silly_IntOrFloat:$rhs
  );

  let results = (outs Silly_IntOrFloat:$result);

  let assemblyFormat = [{
    $kind $lhs `:` type($lhs)
    `,` $rhs `:` type($rhs)
    `->` type($result) attr-dict
  }];
}

def Silly_CmpBinOpKind : I32EnumAttr<"CmpBinOpKind",
                                  "arithmetic comparison operation kind", [
  I32EnumAttrCase<"Less",    0, "less">,
  I32EnumAttrCase<"LessEq",  1, "less_eq">,
  I32EnumAttrCase<"Equal",   2, "eq">,
  I32EnumAttrCase<"NotEqual",3, "neq">,
]> {
  let cppNamespace = "::silly";
}

def Silly_CmpBinOp : Op<Silly_Dialect, "cmp"> {
  let summary = "Generic comparison operation (less, less_eq, etc.)";
  let description = [{
    Performs a comparison selected by the kind attribute.
    Operands must have the same type. Result is always i1 (boolean).
  }];

  let arguments = (ins
    Silly_CmpBinOpKind:$kind,
    Silly_IntOrFloat:$lhs,
    Silly_IntOrFloat:$rhs
  );

  let results = (outs I1:$result);

  let assemblyFormat = [{
    $kind $lhs `:` type($lhs)
    `,` $rhs `:` type($rhs)
    `->` type($result) attr-dict
  }];
}

def Silly_NegOp : Op<Silly_Dialect, "negate"> {
  let summary = "Negation operation (e.g., -x)";
  let arguments = (ins Silly_IntOrFloat:$lhs);
  let results = (outs Silly_IntOrFloat:$result);
  //let assemblyFormat = "`-` $lhs `:` type($lhs) attr-dict";
}

// vim: et ts=2 sw=2
