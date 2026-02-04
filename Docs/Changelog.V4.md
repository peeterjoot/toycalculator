## tag: V4 (July 7, 2025)

The big changes in this tag relative to V3 are:

* Adds support (grammar, builder, lowering) for function declarations, and function calls.  Much of the work for this was done in branch `peeter/old/use_mlir_funcop_with_scopeop`, later squashed and merged as a big commit.
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
      "silly.scope"() ({
        "silly.declare"() <{type = i16}> {sym_name = "v"} : () -> ()
        %c3_i64 = arith.constant 3 : i64
        "silly.assign"(%c3_i64) <{var_name = @v}> : (i64) -> ()
        %0 = "silly.string_literal"() <{value = "In foo"}> : () -> !llvm.ptr
        silly.print %0 : !llvm.ptr
        %1 = "silly.load"() <{var_name = @v}> : () -> i16
        %c42_i64 = arith.constant 42 : i64
        %2 = arith.trunci %c42_i64 : i64 to i32
        "silly.call"(%1, %2) <{callee = @bar}> : (i16, i32) -> ()
        %3 = "silly.string_literal"() <{value = "Called bar"}> : () -> !llvm.ptr
        silly.print %3 : !llvm.ptr
        "silly.return"() : () -> ()
      }) : () -> ()
      "silly.yield"() : () -> ()
    }
    func.func private @bar(%arg0: i16, %arg1: i32) {
      "silly.scope"() ({
        "silly.declare"() <{param_number = 0 : i64, parameter, type = i16}> {sym_name = "w"} : () -> ()
        "silly.declare"() <{param_number = 1 : i64, parameter, type = i32}> {sym_name = "z"} : () -> ()
        %0 = "silly.string_literal"() <{value = "In bar"}> : () -> !llvm.ptr
        silly.print %0 : !llvm.ptr
        %1 = "silly.load"() <{var_name = @w}> : () -> i16
        silly.print %1 : i16
        %2 = "silly.load"() <{var_name = @z}> : () -> i32
        silly.print %2 : i32
        "silly.return"() : () -> ()
      }) : () -> ()
      "silly.yield"() : () -> ()
    }
    func.func @main() -> i32 {
      "silly.scope"() ({
        %c0_i32 = arith.constant 0 : i32
        %0 = "silly.string_literal"() <{value = "In main"}> : () -> !llvm.ptr
        silly.print %0 : !llvm.ptr
        "silly.call"() <{callee = @foo}> : () -> ()
        %1 = "silly.string_literal"() <{value = "Back in main"}> : () -> !llvm.ptr
        silly.print %1 : !llvm.ptr
        "silly.return"(%c0_i32) : (i32) -> ()
      }) : () -> ()
      "silly.yield"() : () -> ()
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
      "silly.scope"() ({
        "silly.declare"() <{param_number = 0 : i64, parameter, type = i16}> {sym_name = "w"} : () -> ()
        %0 = "silly.load"() <{var_name = @w}> : () -> i16
        silly.print %0 : i16
        "silly.return"() : () -> ()
      }) : () -> ()
      "silly.yield"() : () -> ()
    }
    func.func @main() -> i32 {
      "silly.scope"() ({
        %c0_i32 = arith.constant 0 : i32
        %0 = "silly.string_literal"() <{value = "In main"}> : () -> !llvm.ptr
        silly.print %0 : !llvm.ptr
        %c3_i64 = arith.constant 3 : i64
        %1 = arith.trunci %c3_i64 : i64 to i16
        "silly.call"(%1) <{callee = @bar}> : (i16) -> ()
        %2 = "silly.string_literal"() <{value = "Back in main"}> : () -> !llvm.ptr
        silly.print %2 : !llvm.ptr
        "silly.return"(%c0_i32) : (i32) -> ()
      }) : () -> ()
      "silly.yield"() : () -> ()
    }
  }
```

  I've implemented a two stage lowering, where the silly.scope, silly.yield, silly.call, and silly.returns are stripped out leaving just the func and llvm dialects.  Code from that stage of the lowering is cleaner looking

```
  llvm.mlir.global private constant @str_1(dense<[66, 97, 99, 107, 32, 105, 110, 32, 109, 97, 105, 110]> : tensor<12xi8>) {addr_space = 0 : i32} : !llvm.array<12 x i8>
  func.func private @__silly_print_string(i64, !llvm.ptr)
  llvm.mlir.global private constant @str_0(dense<[73, 110, 32, 109, 97, 105, 110]> : tensor<7xi8>) {addr_space = 0 : i32} : !llvm.array<7 x i8>
  func.func private @__silly_print_i64(i64)
  func.func private @bar(%arg0: i16) {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x i16 {alignment = 2 : i64, bindc_name = "w.addr"} : (i64) -> !llvm.ptr
    llvm.store %arg0, %1 : i16, !llvm.ptr
    %2 = llvm.load %1 : !llvm.ptr -> i16
    %3 = llvm.sext %2 : i16 to i64
    call @__silly_print_i64(%3) : (i64) -> ()
    return
  }
  func.func @main() -> i32 {
    %0 = llvm.mlir.constant(0 : i32) : i32
    %1 = llvm.mlir.addressof @str_0 : !llvm.ptr
    %2 = llvm.mlir.constant(7 : i64) : i64
    call @__silly_print_string(%2, %1) : (i64, !llvm.ptr) -> ()
    %3 = llvm.mlir.constant(3 : i64) : i64
    %4 = llvm.mlir.constant(3 : i16) : i16
    call @bar(%4) : (i16) -> ()
    %5 = llvm.mlir.addressof @str_1 : !llvm.ptr
    %6 = llvm.mlir.constant(12 : i64) : i64
    call @__silly_print_string(%6, %5) : (i64, !llvm.ptr) -> ()
    return %0 : i32
  }
```

  There are some dead code constants left there (%3), seeming due to type conversion, but they get stripped out nicely by the time we get to LLVM-IR:

```
  @str_1 = private constant [12 x i8] c"Back in main"
  @str_0 = private constant [7 x i8] c"In main"

  declare void @__silly_print_string(i64, ptr)

  declare void @__silly_print_i64(i64)

  define void @bar(i16 %0) {
    %2 = alloca i16, i64 1, align 2
    store i16 %0, ptr %2, align 2
    %3 = load i16, ptr %2, align 2
    %4 = sext i16 %3 to i64
    call void @__silly_print_i64(i64 %4)
    ret void
  }

  define i32 @main() {
    call void @__silly_print_string(i64 7, ptr @str_0)
    call void @bar(i16 3)
    call void @__silly_print_string(i64 12, ptr @str_1)
    ret i32 0
  }
```

* Generalize NegOp lowering to support all types, not just f64.
* Allow PRINT of string literals, avoiding requirement for variables.  Example:

```
  %0 = "silly.string_literal"() <{value = "A string literal!"}> : () -> !llvm.ptr loc(#loc)
  "silly.print"(%0) : (!llvm.ptr) -> () loc(#loc)
```

  There were lots of internal changes made along the way, one of which ended up reverted:
* Cache constantop values so that they need not be repeated -- that caching should be function specific, and will have to be generalized.


Other internal changes include:

* Generate the __silly_print... prototypes on demand, to clutter up the generated code less.  Can do this by saving and restoring the insertion point to the module level (where the symbol table and globals live.)
* Introduce a string literal op, replacing the customized string assign operator:

```
  silly.string_assign "s" = "hi"
```

with plain old assign, after first constructing a string literal object:

```
  %0 = "silly.string_literal"() <{value = "hi"}> : () -> !llvm.ptr
  silly.assign "s", %0 : !llvm.ptr
```

* Standardize Type handling in lowering.  Cache all the supported int/float types so that I can do compares to those.  This meant that a wide variety of operations, for example:
  * IntegerType::get(...)
  * builder.getI64Type(), ...
  * rewriter.getI64Type(), ...
  * mlir::isa
  * mlir::dyn_cast
  could all be eliminated, replaced with the cached type values of interest.
* Grammar: add ifelifelse rule (samples/if.silly).  No builder nor lowering support yet.
* Lowering: Fix StoreOp alignment (had i64's with align 4 in the generated ll.)
* Replace silly::ProgramOp with mlir::func::FuncOp (prep for adding scopes and callable functions.)
* Grammar now has FUNCTION syntax (assert stub in parser, no builder/lowering yet.)
* Grammar: rename `VARIABLENAME_PATTERN -> IDENTIFIER`
* Parser: intercept errors instead of letting parse tree walker autocorrect and continue.
* New error tests: `error_keyword_declare.silly error_keyword_declare2.silly`
* Split lowering into two passes, with separate pass for FuncOp, so that we have option of keeping function symbol tables through (dcl/assign/load) op lowering.
* Parser now using symbol table anchored to silly::FuncOp, replacing hashes.  lowering still uses a hash, but it's function:: qualified.
* constants.hpp: `ENTRY_SYMBOL_NAME`, ... (avoiding hardcoded duplication.)
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
* Initial builder support for CALL (fails in lowering.)  Tried using mlir::func::CallOp, but that doesn't like my use of silly::FuncOp instead of mlir::func::FuncOp.  I did that so that my function object had a symbol table for local variables, but it looks like a better approach would be to implement a ScopeOp that has the symbol table, and to then embed ScopeOp in a mlir::func::FuncOp region.
* parser: Remove: lastOperator lastOp, and exitStartRule.  Instead put in a dummy exit when the scope is created and replace it later with one that has values if required.
* Replace FuncOp/ExitOp with mlir::func::FuncOp/ReturnOp.
* Add parameter and param_number attrs to DeclareOp, and lower DeclareOp w/ parameter to parameter specific dwarf DI instrumentation.  Lower parameter dcl to alloca+store+dbg.declare
* Purge the 0/1 constantop caching.  That only worked for a single (main) function.  Would have to be more clever to make that work in the general case (recording the function associated with the caching or something like that.)

---

<!-- vim: set tw=80 ts=2 sw=2 et: -->
