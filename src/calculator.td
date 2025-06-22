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
include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Dialect/LLVMIR/LLVMTypes.td"

def Toy_Dialect : Dialect {
  let name = "toy";
  let summary = "A compiler MLIR dialect for a simple calculator with variables";
  let cppNamespace = "toy";
}

def Toy_FuncOp : Op<Toy_Dialect, "func", [FunctionOpInterface]> {
  let summary = "Function operation";
  let description = "A function operation.  Designed to mimic mlir::func::FuncOp, but with a symbol table for local variables.";
  let arguments = (ins StrAttr:$sym_name, TypeAttr:$function_type, OptionalAttr<StrAttr>:$sym_visibility);
  let results = (outs);
  let regions = (region AnyRegion:$body);
  let traits = [SymbolTable, AutomaticAllocationScope,
                //IsIsolatedFromAbove,
                OpAsmOpInterface, Symbol];
  let extraClassDeclaration = [{
    mlir::Block *addEntryBlock() {
      assert(getBody().empty() && "Function already has an entry block!");
      mlir::Block & entry = getBody().emplaceBlock();
      auto type = dyn_cast<mlir::FunctionType>(getFunctionTypeAttr().getValue());
      auto locs = llvm::SmallVector<mlir::Location>(type.getInputs().size(), getLoc());
      entry.addArguments(type.getInputs(), locs);
      return &entry;
    }

    // these three are for mlir::func::FuncOp compatibility -- but may not be needed:
    mlir::FunctionType getFunctionTypeAttrValue() {
      return dyn_cast<mlir::FunctionType>(getFunctionTypeAttr().getValue());
    }

    mlir::StringAttr getSymNameAttribute() {
      return mlir::StringAttr::get(getContext(), getSymName());
    }

    mlir::StringAttr getSymVisibilityAttribute() {
      if (auto vis = getSymVisibility()) {
        return mlir::StringAttr::get(getContext(), *vis);
      }
      return nullptr;
    }

    static void build(::mlir::OpBuilder &builder, ::mlir::OperationState &state,
                     ::llvm::StringRef symName, ::mlir::FunctionType funcType,
                     ::mlir::StringAttr symVisibility = nullptr) {
      state.addAttribute("sym_name", ::mlir::StringAttr::get(state.getContext(), symName));
      state.addAttribute("function_type", ::mlir::TypeAttr::get(funcType));
      if (symVisibility) {
        state.addAttribute("sym_visibility", symVisibility);
      }
      state.addRegion();
    }

    mlir::LogicalResult verify() {
      if (!getSymNameAttr()) {
        return emitOpError("requires a 'sym_name' attribute");
      }
      if (!getFunctionTypeAttr()) {
        return emitOpError("requires a 'function_type' attribute");
      }
      if (!getBody().empty() && getBody().getBlocks().size() != 1) {
        return emitOpError("expects exactly one block in the body region");
      }
      return mlir::success();
    }
  }];
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

def Toy_LocalOp : Op<Toy_Dialect, "local"> {
  let summary = "Intermediate lowering state for a declared local variable.";
  let arguments = (ins LLVMPointerType:$alloca);
  let results = (outs);
  let traits = [Symbol];
  let description = [{
Intermediate lowering state for a declared local variable.

Stage I lowering leaves removes all ToyDialect artifacts other than toy::FuncOp (which holds a symbol table, the region, ...)
and toy::LocalOp (this type.)  This type is not emitted by the original parser/builder pass, which generates DeclareOp for each declare.

The type and size info contained in an original DeclareOp is lowered to AllocaOp, but the sym_name of that DeclareOp is reregistered
with the FuncOp, so that we can do a symbol name -> alloca lookup in LoadOp and AssignOp lowering.

Stage II lowering converts toy::FuncOp to mlir::func::FuncOp, transferring the region from the original, and throwing away the symbol table.
It also lowers toy::LocalOp to the contained AllocA, throwing away any symbol table references.
}];

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

// vim: et ts=2 sw=2
