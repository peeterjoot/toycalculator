//
// @file    silly.md
// @author  Peeter Joot <peeterjoot@pm.me>
// @brief   This implements the Silly compiler dialect.
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

def Silly_Dialect : Dialect {
  let name = "silly";
  let summary = "An MLIR dialect for a silly simple language";
  let cppNamespace = "silly";
}

def Silly_VarType : TypeDef<Silly_Dialect, "var"> {
  let summary = "Abstract variable location (scalar or array)";
  let description = [{
    Represents an abstract handle to a variable's storage.
    Scalars have empty shape; arrays have a non-empty shape.
    This type is storage-agnostic and lowered to concrete memory
    (e.g., LLVM alloca).
  }];

  let parameters = (ins
    "Type":$elementType,
    "llvm::ArrayRef<int64_t>":$shape   // empty = scalar
  );

  let mnemonic = "var";

  let assemblyFormat = "`<` $elementType `[` $shape `]` `>`";
}

def Silly_DeclareOp : Op<Silly_Dialect, "declare"> {
  let summary = "Declare an abstract variable location (scalar or array).";
  let description = [{
    Declares a variable (scalar or array) or function parameter with a name and type.
    For parameters, the `parameter` attribute is set, and `param_number` indicates the parameter index.
    The result is an abstract handle (!silly.var) to the variable's location, storage-agnostic.
    Initializers are provided as operands; for arrays, excess initializers are an error.
  }];
  let arguments = (ins
    Variadic<AnyType>:$initializers,
    OptionalAttr<UnitAttr>:$parameter,
    OptionalAttr<I64Attr>:$param_number,
    StrAttr:$sym_name
  );
  let results = (outs Silly_VarType:$var);
  let traits = [Symbol];

  let assemblyFormat = "$initializers `:` type($initializers) attr-dict `:` type($var)";

  let extraClassDeclaration = [{
    mlir::StringAttr getSymNameAttr() { return getSymName(); }

    bool isParameter() { return getParameterAttr() != nullptr; }

    mlir::LogicalResult verify() {
//      // Symbol name must exist and be non-empty
//      if (getSymName().empty()) {
//        return emitOpError("requires a non-empty 'sym_name' attribute of type StringAttr.");
//      }
//    
//      // Parameter consistency
//      if (isParameter() && !getParamNumberAttr()) {
//        return emitOpError("parameter declarations require a 'param_number' attribute.");
//      }
//      if (!isParameter() && getParamNumberAttr()) {
//        return emitOpError("non-parameter declarations cannot have a 'param_number' attribute.");
//      }
//    
//      // Result type must be !silly.var<...>
//      auto varType = mlir::dyn_cast<silly::varType>(getVar().getType());
//      if (!varType) {
//        return emitOpError("result must be of type !silly.var");
//      }
//    
//      // Determine element count from the type shape
//      llvm::ArrayRef<int64_t> shape = varType.getShape();
//    
//      if (!shape.empty()) {
//        if (shape.size() != 1) {
//          return emitOpError("only 1D arrays are supported (rank must be 0 or 1)");
//        }
//        if (shape[0] <= 0) {
//          return emitOpError("array size must be a positive integer");
//        }
//      }
//    
//      // Check initializer count
//      size_t numInits = getInitializers().size();
//      size_t numElements = shape.empty() ? 1 : shape[0];
//    
//      if (numInits > numElements) {
//        return emitOpError("number of initializers (")
//               << numInits << ") exceeds number of elements (" << numElements << ")";
//      }
//    
//      // Optional: type-check initializers
//      Type elemTy = varType.getElementType();
//      for (Value init : getInitializers()) {
//        if (init.getType() != elemTy) {
//          return emitOpError("initializer type ")
//                 << init.getType()
//                 << " does not match variable element type "
//                 << elemTy;
//        }
//      }
//    
      return mlir::success();
    }

  }];
}

def Silly_IntOrFloat : TypeConstraint<
  CPred<"mlir::isa<mlir::IntegerType>($_self) || mlir::isa<mlir::FloatType>($_self)">,
  "integer or float type">;

def Silly_AssignOp : Op<Silly_Dialect, "assign"> {
  let summary = "Assign a value to a variable (scalar or array element).";

  let description = [{
    Assigns `value` to the variable referenced by `var_name`.
    If `index` is present, the assignment targets the array element at that index.
    The target variable must have been declared with a matching `silly.declare`.
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

def Silly_DebugName : Op<Silly_Dialect, "debug_name", [NoMemoryEffect]> {
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
    SymbolRefAttr:$var_name,               // @t
    Optional<Index>:$index                 // optional SSA value of index type (dynamic or none)
  );

  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$var_name (`[` $index^ `]`)? `:` type($result) attr-dict";
}

def Silly_YieldOp : Op<Silly_Dialect, "yield"> {
  let summary = "Terminator for silly.scope blocks";
  let description = [{
    A no-op terminator operation for `silly.scope` blocks to satisfy MLIR's block
    termination requirement. It carries no semantic meaning and is removed during
    lowering.
  }];
  let arguments = (ins);
  let results = (outs);
  let traits = [Terminator];
}

def Silly_ScopeOp : Op<Silly_Dialect, "scope"> {
  let summary = "A scoping construct for local variables in the Silly dialect";
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

def Silly_ReturnOp : Op<Silly_Dialect, "return"> {
  let summary = "Return operation for silly.scope blocks";
  let description = [{
    A return operation within a `silly.scope` that returns a value to the parent function.
    It is lowered to a `func.return` during `ScopeOp` lowering.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs);
  let traits = [Terminator];
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

def Silly_NegOp : Op<Silly_Dialect, "negate"> {
  let summary = "Negation operation (e.g., -x)";
  let arguments = (ins Silly_IntOrFloat:$lhs);
  let results = (outs Silly_IntOrFloat:$result);
  //let assemblyFormat = "`-` $lhs `:` type($lhs) attr-dict";
}

def Silly_AddOp : Op<Silly_Dialect, "add"> {
  let summary = "Addition operation (e.g., x + y)";
  let arguments = (ins Silly_IntOrFloat:$lhs, Silly_IntOrFloat:$rhs);
  let results = (outs Silly_IntOrFloat:$result);
  //let assemblyFormat = "$lhs `+` $rhs `:` type($result) attr-dict";
}

def Silly_SubOp : Op<Silly_Dialect, "sub"> {
  let summary = "Subtraction operation (e.g., x - y)";
  let arguments = (ins Silly_IntOrFloat:$lhs, Silly_IntOrFloat:$rhs);
  let results = (outs Silly_IntOrFloat:$result);
  //let assemblyFormat = "$lhs `-` $rhs `:` type($result) attr-dict";
}

def Silly_MulOp : Op<Silly_Dialect, "mul"> {
  let summary = "Multiplication operation (e.g., x * y)";
  let arguments = (ins Silly_IntOrFloat:$lhs, Silly_IntOrFloat:$rhs);
  let results = (outs Silly_IntOrFloat:$result);
  //let assemblyFormat = "$lhs `*` $rhs `:` type($result) attr-dict";
}

def Silly_DivOp : Op<Silly_Dialect, "div"> {
  let summary = "Division operation (e.g., x / y)";
  let arguments = (ins Silly_IntOrFloat:$lhs, Silly_IntOrFloat:$rhs);
  let results = (outs Silly_IntOrFloat:$result);
  //let assemblyFormat = "$lhs `/` $rhs `:` type($result) attr-dict";
}

def Silly_LessOp : Op<Silly_Dialect, "less"> {
  let summary = "Less than operation (e.g., x < y)";
  let arguments = (ins Silly_IntOrFloat:$lhs, Silly_IntOrFloat:$rhs);
  let results = (outs I1:$result);
  //let assemblyFormat = "$lhs `<` $rhs `:` type($result) attr-dict";
}

def Silly_LessEqualOp : Op<Silly_Dialect, "less_eq"> {
  let summary = "Less than or equal operation (e.g., x <= y)";
  let arguments = (ins Silly_IntOrFloat:$lhs, Silly_IntOrFloat:$rhs);
  let results = (outs I1:$result);
  //let assemblyFormat = "$lhs `<=` $rhs `:` type($result) attr-dict";
}

def Silly_EqualOp : Op<Silly_Dialect, "eq"> {
  let summary = "Equal operation (e.g., x EQ y)";
  let arguments = (ins Silly_IntOrFloat:$lhs, Silly_IntOrFloat:$rhs);
  let results = (outs I1:$result);
  //let assemblyFormat = "$lhs `==` $rhs `:` type($result) attr-dict";
}

def Silly_NotEqualOp : Op<Silly_Dialect, "neq"> {
  let summary = "Not equal operation (e.g., x NE y)";
  let arguments = (ins Silly_IntOrFloat:$lhs, Silly_IntOrFloat:$rhs);
  let results = (outs I1:$result);
  //let assemblyFormat = "$lhs `!=` $rhs `:` type($result) attr-dict";
}

def Silly_OrOp : Op<Silly_Dialect, "or"> {
  let summary = "Bitwise or boolean OR operation (e.g., x OR y)";
  let arguments = (ins Silly_IntOrFloat:$lhs, Silly_IntOrFloat:$rhs);
  let results = (outs Silly_IntOrFloat:$result);
  //let assemblyFormat = "$lhs `|` $rhs `:` type($result) attr-dict";
}

def Silly_AndOp : Op<Silly_Dialect, "and"> {
  let summary = "Bitwise or boolean AND operation (e.g., x AND y)";
  let arguments = (ins Silly_IntOrFloat:$lhs, Silly_IntOrFloat:$rhs);
  let results = (outs Silly_IntOrFloat:$result);
  //let assemblyFormat = "$lhs `&` $rhs `:` type($result) attr-dict";
}

def Silly_XorOp : Op<Silly_Dialect, "xor"> {
  let summary = "Bitwise or boolean XOR operation (e.g., x XOR y)";
  let arguments = (ins Silly_IntOrFloat:$lhs, Silly_IntOrFloat:$rhs);
  let results = (outs Silly_IntOrFloat:$result);
  //let assemblyFormat = "$lhs `^` $rhs `:` type($result) attr-dict";
}

// vim: et ts=2 sw=2
