## tag: V5 (WIP)

* Fix Location info for implicit returns, using the last function location info.

## tag: V4 (July 7, 2025)

The big changes in this tag relative to V3 are:
* Adds support (grammar, builder, lowering) for function declarations, and function calls.  Much of the work for this was done in branch use_mlir_funcop_with_scopeop, later squashed and merged as a big commit.
Here's an example

```
FUNCTION bar ( INT16 w, INT32 z )
{
    PRINT "In bar";
    PRINT w;
    PRINT z;
    RETURN;
};

FUNCTION foo ( )
{
    INT16 v;
    v = 3;
    PRINT "In foo";
    CALL bar( v, 42 );
    PRINT "Called bar";
    RETURN;
};

PRINT "In main";
CALL foo();
PRINT "Back in main";
```

Here is the MLIR for this program:

```
module {
  func.func private @foo() {
    "toy.scope"() ({
      "toy.declare"() <{type = i16}> {sym_name = "v"} : () -> ()
      %c3_i64 = arith.constant 3 : i64
      "toy.assign"(%c3_i64) <{var_name = @v}> : (i64) -> ()
      %0 = "toy.string_literal"() <{value = "In foo"}> : () -> !llvm.ptr
      toy.print %0 : !llvm.ptr
      %1 = "toy.load"() <{var_name = @v}> : () -> i16
      %c42_i64 = arith.constant 42 : i64
      %2 = arith.trunci %c42_i64 : i64 to i32
      "toy.call"(%1, %2) <{callee = @bar}> : (i16, i32) -> ()
      %3 = "toy.string_literal"() <{value = "Called bar"}> : () -> !llvm.ptr
      toy.print %3 : !llvm.ptr
      "toy.return"() : () -> ()
    }) : () -> ()
    "toy.yield"() : () -> ()
  }
  func.func private @bar(%arg0: i16, %arg1: i32) {
    "toy.scope"() ({
      "toy.declare"() <{param_number = 0 : i64, parameter, type = i16}> {sym_name = "w"} : () -> ()
      "toy.declare"() <{param_number = 1 : i64, parameter, type = i32}> {sym_name = "z"} : () -> ()
      %0 = "toy.string_literal"() <{value = "In bar"}> : () -> !llvm.ptr
      toy.print %0 : !llvm.ptr
      %1 = "toy.load"() <{var_name = @w}> : () -> i16
      toy.print %1 : i16
      %2 = "toy.load"() <{var_name = @z}> : () -> i32
      toy.print %2 : i32
      "toy.return"() : () -> ()
    }) : () -> ()
    "toy.yield"() : () -> ()
  }
  func.func @main() -> i32 {
    "toy.scope"() ({
      %c0_i32 = arith.constant 0 : i32
      %0 = "toy.string_literal"() <{value = "In main"}> : () -> !llvm.ptr
      toy.print %0 : !llvm.ptr
      "toy.call"() <{callee = @foo}> : () -> ()
      %1 = "toy.string_literal"() <{value = "Back in main"}> : () -> !llvm.ptr
      toy.print %1 : !llvm.ptr
      "toy.return"(%c0_i32) : (i32) -> ()
    }) : () -> ()
    "toy.yield"() : () -> ()
  }
}
```

Here's a sample program with an assigned CALL value:

```
FUNCTION bar ( INT16 w )
{
    PRINT w;
    RETURN;
};

PRINT "In main";
CALL bar( 3 );
PRINT "Back in main";
```

The MLIR for this one looks like:

```
module {
  func.func private @bar(%arg0: i16) {
    "toy.scope"() ({
      "toy.declare"() <{param_number = 0 : i64, parameter, type = i16}> {sym_name = "w"} : () -> ()
      %0 = "toy.load"() <{var_name = @w}> : () -> i16
      toy.print %0 : i16
      "toy.return"() : () -> ()
    }) : () -> ()
    "toy.yield"() : () -> ()
  }
  func.func @main() -> i32 {
    "toy.scope"() ({
      %c0_i32 = arith.constant 0 : i32
      %0 = "toy.string_literal"() <{value = "In main"}> : () -> !llvm.ptr
      toy.print %0 : !llvm.ptr
      %c3_i64 = arith.constant 3 : i64
      %1 = arith.trunci %c3_i64 : i64 to i16
      "toy.call"(%1) <{callee = @bar}> : (i16) -> ()
      %2 = "toy.string_literal"() <{value = "Back in main"}> : () -> !llvm.ptr
      toy.print %2 : !llvm.ptr
      "toy.return"(%c0_i32) : (i32) -> ()
    }) : () -> ()
    "toy.yield"() : () -> ()
  }
}
```

I've implemented a two stage lowering, where the toy.scope, toy.yield, toy.call, and toy.returns are stripped out leaving just the func and llvm dialects.  Code from that stage of the lowering is cleaner looking

```
llvm.mlir.global private constant @str_1(dense<[66, 97, 99, 107, 32, 105, 110, 32, 109, 97, 105, 110]> : tensor<12xi8>) {addr_space = 0 : i32} : !llvm.array<12 x i8>
func.func private @__toy_print_string(i64, !llvm.ptr)
llvm.mlir.global private constant @str_0(dense<[73, 110, 32, 109, 97, 105, 110]> : tensor<7xi8>) {addr_space = 0 : i32} : !llvm.array<7 x i8>
func.func private @__toy_print_i64(i64)
func.func private @bar(%arg0: i16) {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.alloca %0 x i16 {alignment = 2 : i64, bindc_name = "w.addr"} : (i64) -> !llvm.ptr
  llvm.store %arg0, %1 : i16, !llvm.ptr
  %2 = llvm.load %1 : !llvm.ptr -> i16
  %3 = llvm.sext %2 : i16 to i64
  call @__toy_print_i64(%3) : (i64) -> ()
  return
}
func.func @main() -> i32 {
  %0 = llvm.mlir.constant(0 : i32) : i32
  %1 = llvm.mlir.addressof @str_0 : !llvm.ptr
  %2 = llvm.mlir.constant(7 : i64) : i64
  call @__toy_print_string(%2, %1) : (i64, !llvm.ptr) -> ()
  %3 = llvm.mlir.constant(3 : i64) : i64
  %4 = llvm.mlir.constant(3 : i16) : i16
  call @bar(%4) : (i16) -> ()
  %5 = llvm.mlir.addressof @str_1 : !llvm.ptr
  %6 = llvm.mlir.constant(12 : i64) : i64
  call @__toy_print_string(%6, %5) : (i64, !llvm.ptr) -> ()
  return %0 : i32
}
```

There are some dead code constants left there (%3), seeming due to type conversion, but they get stripped out nicely by the time we get to LLVM-IR:

```
@str_1 = private constant [12 x i8] c"Back in main"
@str_0 = private constant [7 x i8] c"In main"

declare void @__toy_print_string(i64, ptr)

declare void @__toy_print_i64(i64)

define void @bar(i16 %0) {
  %2 = alloca i16, i64 1, align 2
  store i16 %0, ptr %2, align 2
  %3 = load i16, ptr %2, align 2
  %4 = sext i16 %3 to i64
  call void @__toy_print_i64(i64 %4)
  ret void
}

define i32 @main() {
  call void @__toy_print_string(i64 7, ptr @str_0)
  call void @bar(i16 3)
  call void @__toy_print_string(i64 12, ptr @str_1)
  ret i32 0
}
```

* Generalize NegOp lowering to support all types, not just f64.
* Allow PRINT of string literals, avoiding requirement for variables.  Example:

```
    %0 = "toy.string_literal"() <{value = "A string literal!"}> : () -> !llvm.ptr loc(#loc)
    "toy.print"(%0) : (!llvm.ptr) -> () loc(#loc)
```

There were lots of internal changes made along the way, one of which ended up reverted:
* Cache constantop values so that they need not be repeated -- that caching should be function specific, and will have to be generalized.

Other internal changes include:
* Generate the __toy_print... prototypes on demand, to clutter up the generated code less.  Can do this by saving and restoring the insertion point to the module level (where the symbol table and globals live.)
* Introduce a string literal op, replacing the customized string assign operator:

```
    toy.string_assign "s" = "hi"
```

with plain old assign, after first constructing a string literal object:

```
    %0 = "toy.string_literal"() <{value = "hi"}> : () -> !llvm.ptr
    toy.assign "s", %0 : !llvm.ptr
```

* Standardize Type handling in lowering.  Cache all the supported int/float types so that I can do compares to those.  This meant that a wide variety of operations, for example:
  - IntegerType::get(...)
  - builder.getI64Type(), ...
  - rewriter.getI64Type(), ...
  - mlir::isa
  - mlir::dyn_cast
  could all be eliminated, replaced with the cached type values of interest.
* Grammar: add ifelifelse rule (samples/if.toy).  No builder nor lowering support yet.
* Lowering: Fix StoreOp alignment (had i64's with align 4 in the generated ll.)
* Replace toy::ProgramOp with mlir::func::FuncOp (prep for adding scopes and callable functions.)
* Grammar now has FUNCTION syntax (assert stub in parser, no builder/lowering yet.)
* Grammar: rename VARIABLENAME_PATTERN -> IDENTIFIER
* Parser: intercept errors instead of letting parse tree walker autocorrect and continue.
* New error tests: error_keyword_declare.toy error_keyword_declare2.toy
* Split lowering into two passes, with separate pass for FuncOp, so that we have option of keeping function symbol tables through (dcl/assign/load) op lowering.
* Parser now using symbol table anchored to toy::FuncOp, replacing hashes.  lowering still uses a hash, but it's function:: qualified.
* constants.hpp: ENTRY_SYMBOL_NAME, ... (avoiding hardcoded duplication.)
* Refactor "main" DI instrumentation for generic function support, and generalize the !DISubroutineType creation logic for user defined functions.
* Introduce useModuleInsertionPoint to save and restore the insertion point to the module body (lowering)
* Until ready to support premature return (when control flow possibilities are allowed), have enforced mandatory RETURN at function end in the grammar.
* Add parser support for variable declarations in different functions.
* Implement enterFunction, exitFunction, enterReturnStatement
* Fix statement/returnStatement parse ambiguity.  statement was too greedy, including returnStatement
* Handle save/restore insertion point for user defined functions
* Lowering for void return (hack: may split EXIT/RETURN lowering.)
* Parser support for functions with non-void return/params.
* Grammar support for CALL(...) and assignment 'x = CALL FOO(...)'
* Initial builder support for CALL (fails in lowering.)  Tried using mlir::func::CallOp, but that doesn't like my use of Toy::FuncOp instead of mlir::func::FuncOp.  I did that so that my function object had a symbol table for local variables, but it looks like a better approach would be to implement a ScopeOp that has the symbol table, and to then embed ScopeOp in a mlir::func::FuncOp region.
* parser: Remove: lastOperator lastOp, and exitStartRule.  Instead put in a dummy exit when the scope is created and replace it later with one that has values if required.
* Replace FuncOp/ExitOp with mlir::func::FuncOp/ReturnOp.
* Add parameter and param_number attrs to DeclareOp, and lower DeclareOp w/ parameter to parameter specific dwarf DI instrumentation.  Lower parameter dcl to alloca+store+dbg.declare
* Purge the 0/1 constantop caching.  That only worked for a single (main) function.  Would have to be more clever to make that work in the general case (recording the function associated with the caching or something like that.)

## tag: V3 (Jun 2, 2025)

LANGUAGE ELEMENTS:
* comparison operators (<, <=, EQ, NE) yielding BOOL values.  These work for any combinations of floating and integer types (including BOOL.)
* integer bitwise operators (OR, AND, XOR).  These only for for integer types (including BOOL.)
* a NOT operator, yielding BOOL.
* Array + string declaration and lowering support, including debug instrumentation, and print support for string variables.
* String assignment support.

TEST:
* move samples/testit.sh to bin/testit
* testit: Document --optimize.  Add --assembly, --no-debug
* test case generators for all the boolean and bitwise operations.
* many new tests.

INTERNALS:
* Fixed -g/-OX propagation to lowering.  If -g not specified, now don't generate the DI.
* Show the optimized .ll with --emit-llvm instead of the just-lowered .ll (unless not invoking the assembly printer, where the ll optimization passes are registered.)
* Reorganize the grammar so that all the simple lexer tokens are last.  Rename a bunch of the tokens, introducing some consistency.
* calculator.td: introduce IntOrFloat constraint type, replacing AnyType usage; array decl support, and string support.
* driver: writeLL helper function, pass -g to lowering if set.
* parser: handle large integer constants properly, array decl support, and string support.
* simplest.cpp: This MWE is updated to include a global variable and global variable access.
* parser: implicit exit: use the last saved location, instead of the module start.  This means the line numbers don't jump around at the very end of the program anymore (i.e.: implicit return/exit)

## tag: V2 (May 25, 2025)

This release:

* Adds DWARF debug instrumentation.  Yay!
* Builds a link step into the compiler driver.  To avoid that, compile with -c.

## tag: V1 (May 17, 2025)

* Declare variables with BOOL, INT8, INT16, INT32, INT64, FLOAT32, FLOAT64 types:

```
BOOL b;
INT16 i;
FLOAT32 f;
```

* TRUE, FALSE, and floating point constants:

```
b = TRUE;
f = 5 + 3.14E0;
```

* An EXIT builtin to return a Unix command line value (must be the last statement in the program):

```
EXIT 1;
EXIT x;

```
* Expression type conversions:

```
INT16 x;
FLOAT32 y;
y = 3.14E0;
x = 1 + y;
```

The type conversion rules in the language are not like C.
Instead, all expression elements are converted to the type of the destination before the operation, and integers are truncated.
Example:

```
INT32 x;
x = 1.78 + 3.86E0;
FLOAT64 f;
f = x;
PRINT f;
f = 1.78 + 3.86E0;
PRINT f;
```

The expected output for this program is:

```
4.000000
5.640000
```

### MLIR examples

The MLIR for the language now matches the statements of the language much more closely.  Consider test.toy for example:

```
DCL x;
x = 5 + 3.14E0;
PRINT x;
DCL y;
y = x * 2;
PRINT y;
```

for which the MLIR is now free of memref dialect:

```
"builtin.module"() ({
  "toy.program"() ({
    "toy.declare"() <{name = "x", type = f64}> : () -> () loc(#loc)
    %0 = "arith.constant"() <{value = 5 : i64}> : () -> i64 loc(#loc1)
    %1 = "arith.constant"() <{value = 3.140000e+00 : f64}> : () -> f64 loc(#loc1)
    %2 = "toy.add"(%0, %1) : (i64, f64) -> f64 loc(#loc1)
    "toy.assign"(%2) <{name = "x"}> : (f64) -> () loc(#loc1)
    %3 = "toy.load"() <{name = "x"}> : () -> f64 loc(#loc2)
    "toy.print"(%3) : (f64) -> () loc(#loc2)
    "toy.declare"() <{name = "y", type = f64}> : () -> () loc(#loc3)
    %4 = "toy.load"() <{name = "x"}> : () -> f64 loc(#loc4)
    %5 = "arith.constant"() <{value = 2 : i64}> : () -> i64 loc(#loc4)
    %6 = "toy.mul"(%4, %5) : (f64, i64) -> f64 loc(#loc4)
    "toy.assign"(%6) <{name = "y"}> : (f64) -> () loc(#loc4)
    %7 = "toy.load"() <{name = "y"}> : () -> f64 loc(#loc5)
    "toy.print"(%7) : (f64) -> () loc(#loc5)
    "toy.exit"() : () -> () loc(#loc)
  }) : () -> () loc(#loc)
}) : () -> () loc(#loc)
#loc = loc("test.toy":1:1)
#loc1 = loc("test.toy":2:5)
#loc2 = loc("test.toy":3:1)
#loc3 = loc("test.toy":4:1)
#loc4 = loc("test.toy":5:5)
#loc5 = loc("test.toy":6:1)
```

I'm still using llvm.alloca, but that now doesn't show up until lowering:

```
; ModuleID = 'test.toy'
source_filename = "test.toy"

declare void @__toy_print(double)

define i32 @main() {
  %1 = alloca double, i64 1, align 8
  store double 8.140000e+00, ptr %1, align 8
  %2 = load double, ptr %1, align 8
  call void @__toy_print(double %2)
  %3 = alloca double, i64 1, align 8
  %4 = load double, ptr %1, align 8
  %5 = fmul double %4, 2.000000e+00
  store double %5, ptr %3, align 8
  %6 = load double, ptr %3, align 8
  call void @__toy_print(double %6)
  ret i32 0
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
```

Example of the generated assembly code for this program:

```
0000000000000000 <main>:
   0:   push   %rax
   1:   movsd  0x0(%rip),%xmm0        # 9 <main+0x9>
   9:   call   e <main+0xe>
   e:   movsd  0x0(%rip),%xmm0        # 16 <main+0x16>
  16:   call   1b <main+0x1b>
  1b:   xor    %eax,%eax
  1d:   pop    %rcx
  1e:   ret
```

## tag: V0

Language elements:

* Declare a double equivalent variable:

```
DCL variablename;
```

* Unary assignments to a variable (constants or plus or minus variable-name):

```
x = 3;
x = +x;
x = -x;
```

* Rudimentary binary operations:

```
x = 5 + 3;
y = x * 2;
```

* A PRINT builtin.

### MLIR examples

This version of the compiler used the memref dialect.  Example:

``` 
> ./build/toycalculator  samples/foo.toy  -g
"builtin.module"() ({
  "toy.program"() ({
    %0 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<f64> loc(#loc1)
    "toy.declare"() <{name = "x"}> : () -> () loc(#loc1)
    %1 = "arith.constant"() <{value = 3 : i64}> : () -> i64 loc(#loc2)
    %2 = "toy.unary"(%1) <{op = "+"}> : (i64) -> f64 loc(#loc2)
    "memref.store"(%2, %0) : (f64, memref<f64>) -> () loc(#loc2)
    "toy.assign"(%2) <{name = "x"}> : (f64) -> () loc(#loc3)
    "toy.print"(%0) : (memref<f64>) -> () loc(#loc4)
    "toy.return"() : () -> () loc(#loc1)
  }) : () -> () loc(#loc1)
}) : () -> () loc(#loc)
#loc = loc("../samples/foo.toy":1:1)
#loc1 = loc("../samples/foo.toy":1:1)
#loc2 = loc("../samples/foo.toy":2:5)
#loc3 = loc("../samples/foo.toy":2:1)
#loc4 = loc("../samples/foo.toy":4:6)
```

That was removed in V1, which now uses a MLIR dialect that matches the language more closely, deferring alloca to lowering.
