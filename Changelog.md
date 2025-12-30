## tag: V7 (WIP)

### 1. Minor maintainance:

* arith pass lowering removed from first pass, as we end up with more after the scf lowering.
* samples/elif.silly: code tests for both paths.
* Split LoweringContext declaration and implementation.  Some stuff made private.
* Fix shortstring2 t/c (print of empty string literal.)
* Fix shortstring3 t/c (assignment of empty string literal).

    Ended up in the non-string literal codepath:

```
  (gdb) up
#11 0x00000000005065d0 in silly::MLIRListener::enterRhs (this=0x7fffffffd7d8, ctx=0x7296d0) at /home/pjoot/toycalculator/src/parser.cpp:1732
1732                    builder.create<silly::AssignOp>( loc, mlir::TypeRange{}, mlir::ValueRange{ resultValue },
```

    checking s.length() isn't appropriate.
* Added integer literal support to PRINT (t/c: printlit.silly), allowing for a program as simple as:

```
PRINT 42;
```

MLIR:
```
module {
  func.func @main() -> i32 {
    "silly.scope"() ({
      %c42_i64 = arith.constant 42 : i64
      silly.print %c42_i64 : i64
      %cst = arith.constant 4.200000e+01 : f64
      silly.print %cst : f64
      %c0_i32 = arith.constant 0 : i32
      "silly.return"(%c0_i32) : (i32) -> ()
    }) : () -> ()
    "silly.yield"() : () -> ()
  }
}
```

LLVM-LL:
```
declare void @__silly_print_f64(double)

declare void @__silly_print_i64(i64)

define i32 @main() !dbg !4 {
  call void @__silly_print_i64(i64 42), !dbg !8
  call void @__silly_print_f64(double 4.200000e+01), !dbg !9
  ret i32 0, !dbg !9
}
```


## tag: V6 (Dec 28, 2025)

### 1. Get statement support.

* printOp lowering: Merge bulk of createSillyPrintF64Prototype, createSillyPrintI64Prototype functions into a single generic helper function.
* GetOp: Grammar, builder, runtime, and prototype generator helper functions for lowering, and lowering implementation.  Supports BOOL, INT8, INT16, INT32, INT64, FLOAT32, FLOAT64.
* Add test case coverage for each supported type.

### 2. Preliminary FOR statement support.

Example syntax:

```
    FOR ( x : (1, 11) )
    {
        PRINT x;
    };

    FOR ( x : (1, 11, 2) )
    {
        PRINT x;
    };
```

these are equivalent, respectively, to C for loops like:

```
    for ( x = 1 ; x <= 10 ; x += 1 ) { ... }

    for ( x = 1 ; x <= 10 ; x += 2 ) { ... }
```

MLIR for that sample program:

``` 
    module {
      func.func @main() -> i32 {
        "silly.scope"() ({
          "silly.declare"() <{type = i32}> {sym_name = "x"} : () -> ()
          %c1_i64 = arith.constant 1 : i64
          %c11_i64 = arith.constant 11 : i64
          %c1_i64_0 = arith.constant 1 : i64
          scf.for %arg0 = %c1_i64 to %c11_i64 step %c1_i64_0  : i64 {
            silly.assign @x = %arg0 : i64
            %0 = silly.load @x : i32
            silly.print %0 : i32
          }
          %c1_i64_1 = arith.constant 1 : i64
          %c11_i64_2 = arith.constant 11 : i64
          %c2_i64 = arith.constant 2 : i64
          scf.for %arg0 = %c1_i64_1 to %c11_i64_2 step %c2_i64  : i64 {
            silly.assign @x = %arg0 : i64
            %0 = silly.load @x : i32
            silly.print %0 : i32
          }
          %c0_i32 = arith.constant 0 : i32
          "silly.return"(%c0_i32) : (i32) -> ()
        }) : () -> ()
        "silly.yield"() : () -> ()
      }
    }
```

Observe that there's a hack in there: I've inserted a `silly.assign` from the scf.for loop induction variable at the beginning of the loop, so that subsequent symbol based lookups just work.  It would be cleaner to make the FOR loop variable private to the loop body (and have the builder reference the SSA induction variable directly (`forOp.getRegion().front().getArgument(0)`), instead of requiring a variable in the enclosing scope, but I did it this way to avoid the need for any additional dwarf instrumentation for that variable -- basically, I was being lazy, and letting implementation guide the language "design".

My LL looks like:

``` 
    define i32 @main()
      %1 = alloca i32, i64 1, align 4,
        #dbg_declare(ptr %1, !9, !DIExpression(), !8)
      br label %2,
     
    2:                                                ; preds = %5, %0
      %3 = phi i64 [ 1, %0 ], [ %8, %5 ],
      %4 = icmp slt i64 %3, 11,
      br i1 %4, label %5, label %9,
     
    5:                                                ; preds = %2
      %tmp1 = trunc i64 %3 to i32
      store i32 %tmp1, ptr %1, align 4,
      %6 = load i32, ptr %1, align 4,
      %7 = sext i32 %6 to i64,
      call void @__silly_print_i64(i64 %7),
      %8 = add i64 %3, 1,
      br label %2,
     
    9:                                                ; preds = %2
      br label %10,
     
    10:                                               ; preds = %13, %9
      %11 = phi i64 [ 1, %9 ], [ %16, %13 ],
      %12 = icmp slt i64 %11, 11,
      br i1 %12, label %13, label %17,
     
    13:                                               ; preds = %10
      %tmp = trunc i64 %11 to i32
      store i32 %tmp, ptr %1, align 4,
      %14 = load i32, ptr %1, align 4,
      %15 = sext i32 %14 to i64,
      call void @__silly_print_i64(i64 %15),
      %16 = add i64 %11, 2,
      br label %10,
     
    17:                                               ; preds = %10
      ret i32 0,
     
    ; uselistorder directives
      uselistorder ptr %1, { 2, 3, 0, 1 }
      uselistorder i64 %3, { 1, 0, 2 }
      uselistorder i64 %11, { 1, 0, 2 }
    }
```
     
(with !dbg stripped out for clarity, but that sed left inappropriate trailing commas above.)
     
the generated asm is fairly clean, even without optimization:

``` 
    0000000000000000 <main>:
       0:   push   %rbx
       1:   sub    $0x10,%rsp
       5:   mov    $0x1,%ebx
       a:   cmp    $0xa,%rbx
       e:   jg     25 <main+0x25>
      10:   mov    %ebx,0xc(%rsp)
      14:   movslq %ebx,%rdi
      17:   call   1c <main+0x1c>
                            18: R_X86_64_PLT32      __silly_print_i64-0x4
      1c:   inc    %rbx
      1f:   cmp    $0xa,%rbx
      23:   jle    10 <main+0x10>
      25:   mov    $0x1,%ebx
      2a:   cmp    $0xa,%rbx
      2e:   jg     46 <main+0x46>
      30:   mov    %ebx,0xc(%rsp)
      34:   movslq %ebx,%rdi
      37:   call   3c <main+0x3c>
                            38: R_X86_64_PLT32      __silly_print_i64-0x4
      3c:   add    $0x2,%rbx
      40:   cmp    $0xa,%rbx
      44:   jle    30 <main+0x30>
      46:   xor    %eax,%eax
      48:   add    $0x10,%rsp
      4c:   pop    %rbx
      4d:   ret
```
     
With optimization, the loops end up fully unrolled:

``` 
    0000000000000000 <main>:
       0: push %rax
       1: mov $0x1,%edi
       6: call b <main+0xb>
    7: R_X86_64_PLT32 __silly_print_i64-0x4
       b: mov $0x2,%edi
      10: call 15 <main+0x15>
    11: R_X86_64_PLT32 __silly_print_i64-0x4
      15: mov $0x3,%edi
      1a: call 1f <main+0x1f>
    1b: R_X86_64_PLT32 __silly_print_i64-0x4
      1f: mov $0x4,%edi
      24: call 29 <main+0x29>
    25: R_X86_64_PLT32 __silly_print_i64-0x4
      29: mov $0x5,%edi
      2e: call 33 <main+0x33>
    2f: R_X86_64_PLT32 __silly_print_i64-0x4
      33: mov $0x6,%edi
      38: call 3d <main+0x3d>
    39: R_X86_64_PLT32 __silly_print_i64-0x4
      3d: mov $0x7,%edi
      42: call 47 <main+0x47>
    43: R_X86_64_PLT32 __silly_print_i64-0x4
      47: mov $0x8,%edi
      4c: call 51 <main+0x51>
    4d: R_X86_64_PLT32 __silly_print_i64-0x4
      51: mov $0x9,%edi
      56: call 5b <main+0x5b>
    57: R_X86_64_PLT32 __silly_print_i64-0x4
      5b: mov $0xa,%edi
      60: call 65 <main+0x65>
    61: R_X86_64_PLT32 __silly_print_i64-0x4
      65: mov $0x1,%edi
      6a: call 6f <main+0x6f>
    6b: R_X86_64_PLT32 __silly_print_i64-0x4
      6f: mov $0x3,%edi
      74: call 79 <main+0x79>
    75: R_X86_64_PLT32 __silly_print_i64-0x4
      79: mov $0x5,%edi
      7e: call 83 <main+0x83>
    7f: R_X86_64_PLT32 __silly_print_i64-0x4
      83: mov $0x7,%edi
      88: call 8d <main+0x8d>
    89: R_X86_64_PLT32 __silly_print_i64-0x4
      8d: mov $0x9,%edi
      92: call 97 <main+0x97>
    93: R_X86_64_PLT32 __silly_print_i64-0x4
      97: xor %eax,%eax
      99: pop %rcx
      9a: ret
```
 
(which surprised me slightly, but the unrolling is "only" a 2x increase in code size, so I guess it's in the allowable range.)

### 3. Temp switcheroo of the insertion point for all dcls to the begining of the enclosing scopeop.

This fixes ifdcl.silly:

```
INT32 x;

x = 3;

IF ( x < 4 )
{
  INT32 y;
  y = 42;
  PRINT y;
};

PRINT "Done.";
```

which previously failed with y not declared at the assignment point (since the declaration needs the symbol table, which is associated with the ScopeOp)

### 4. Maintainance:
* test:
    samples/testerrors.sh -> bin/testerrors

    ci/cd is now effectively:
```
    cd samples
    testit
    testerrors
```

* parser:
    - switch to `throw exception_with_context` exclusively for errors (no more asserts other than null pointer checks before dereferences.)
    - buildUnaryExpression.  return the value, instead of pass by reference.
    - buildNonStringUnaryExpression.  New helper function.  Just does the no-string-liternal assertion like check (now throw.)
    - various: add formatLocation( loc ) into the thrown error message where possible.
    - purge auto usage.
    - add asserts before any pointer dereferences
    - convert parser:stripQuotes asserts into throw with context
    - doxygen for parser.hpp
    - make formatLocation const.
    - pass loc down to getFuncOp, getEnclosingScopeOp, and lookupDeclareForVar.
    - getFuncOp/getEnclosingScopeOp: throw if not found.
    - buildUnaryExpression: pass loc as first arg, like most other places.
    - Remove dead code: theTypes, getCompilerType, isBoolean, isInteger, isFloat
    - parser.hpp: move inlines out of class dcl for clarity and put public first.
    - parser.cpp: put all the inlines first.  Uninline a few things.
    - remove a bunch of old if-0'd out `LLVM_DEBUG` code.
    - throw `user_error` for syntax errors, and `exception_with_context` only for internal errors. all the enter/exit callbacks catch `user_error` (all for consistency, even if they don't need to, settting hasErrors and calling mlir::emitError(loc) to bubble the error up to the user.)

* Remove `HACK_BUILDER` code.
* Remove constants.hpp.
* bump `COMPILER_VERSION` to V6, matching WIP TAG
* Split out shortstring2.silly and longstring.silly from shortstring.silly.  Added longstring.silly to regression test.
* driver: remove autos
* lowering: remove most autos
* lowering: remove `using namespace mlir` (was using mlir:: qualified variables in some places, but not others, and it was confusing looking.)
* merge testerrors and testit
* `s/exception_with_context/ExceptionWithContext/` ; `s/user_error/UserError/`
* s/driverState/DriverState/
* Remove: prototypes/hibye.  This used the mlir::cf dialect, and I ended up using mlir::scf.
* rename all the underscore variables (close to a consistent convention now.)
* lowering: s/loweringContext/LoweringContext/g;

### 5. Rebranding: toy calculator TO silly compiler/language.

* There's a MLIR tutorial with a toy dialect.  Rename mine so that it can't be confused with something already called toy.

## tag: V5 (Dec 22, 2025)

The language now supports functions, calls, parameters, returns, and basic conditional blocks.

### 1. Build / LLVM Version Updates

* Updated build scripts (`bin/build`, `bin/env`) for LLVM path handling.
* Switched project to LLVM/MLIR 21.x (`llvmorg-21.1.0-rc3` and later). Dropped support for ≤ 20.1.8.
* Added a Flang-related patch (`llvm-patches/llvm21.flang.patch`).

### 2. Return / Location Fixes

* Improved location info for implicit/dummy returns (now uses last statement location instead of file start).
* Introduced per-function parser state to track last location.

### 3. Conditional Statements – IF / ELSE

* Preliminary implementation of IF and ELSE grammar and lowering to `scf.if` / `scf.else`.
* Split the old combined `ifelifelse` rule into separate `if`, `elif`, `else` rules (`ELIF` not yet implemented).
* Added `CallOpLowering` (moved calls out of `ScopeOpLowering` because calls like implicit `PRINT` can now appear inside `if/else` blocks).
* Integrated SCF lowering into the second lowering pass.
* Generalized location helpers and predicate parsing in preparation for full conditional support.
* Updated README to note that IF/ELSE is now supported (but needs much more testing, especially nested IFs).
* Reduced `samples/if.silly` to only the currently implemented subset; moved unimplemented parts to `if2.silly`.

### 4. Array element access and assignment (rvalues and lvalues)

* Added support for array element assignment: `t[i] = expr;`
  - Generalized `silly.assign` to take an optional `index` operand (`Optional<Index>`).
  - Updated grammar to allow `scalarOrArrayElement` (variable optionally indexed) on the LHS of assignments.
  - Implemented lowering of indexed `silly.assign` using `llvm.getelementptr` + `store`, with static out-of-bounds checking for constant indices.
  - Added custom assembly format: `silly.assign @t[%index] = %value : type`.

* Added support for loading array elements (rvalues): `x = t[i];`
  - Generalized `silly.load` to take an optional `index` operand.
  - Implemented lowering using `llvm.getelementptr` + `load`, with static bounds checking.
  - Added custom assembly format: `silly.load @t[%index] : element_type` (scalar case prints without brackets).

* Parser and frontend changes:
  - Introduced `scalarOrArrayElement` and `indexExpression` grammar rules.
  - Grammar: Extended unary expression handling and many statement types (PRINT, RETURN, EXIT, call arguments, etc.) to accept array element references.
  - PRINT/RETURN/EXIT of array elements is supported and tested, but lots more testing should be done in other contexts (call arguments, ...).  It is possible that additional builder/lowering work will show up from such testing.
  - Added `indexTypeCast` helper for converting parsed index values to `index` type.
  - Updated `intarray.silly` sample to demonstrate full round-trip (declare → assign element → load element → print).  Also added rudimentary test for unary and binary expressions with array elements, and exit.
  - Updated `function_intret_void.silly` with return of array element.
  - New test `exitarrayelement.silly` to test exit with non-zero array element (zero tested in `intarray.silly`)

* Lowering improvements:
  - Factored out `castToElemType` helper for consistent type conversion during stores/loads.
  - Fixed several bugs during iterative development (GEP indexing, type handling, optional operand creation).

* README and TODO updates:
  - README now reflects full array element support and notes remaining limitations (no direct printing of array elements, no loops yet).
  - TODO updated with future array enhancements (runtime bounds checking, richer index expressions, element printing).

* Added test case `error_intarray_bad_constaccess.silly` (currently commented in testerrors.sh – static bounds checking catches the error at compile time).

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
  - IntegerType::get(...)
  - builder.getI64Type(), ...
  - rewriter.getI64Type(), ...
  - mlir::isa
  - mlir::dyn_cast
  could all be eliminated, replaced with the cached type values of interest.
* Grammar: add ifelifelse rule (samples/if.silly).  No builder nor lowering support yet.
* Lowering: Fix StoreOp alignment (had i64's with align 4 in the generated ll.)
* Replace silly::ProgramOp with mlir::func::FuncOp (prep for adding scopes and callable functions.)
* Grammar now has FUNCTION syntax (assert stub in parser, no builder/lowering yet.)
* Grammar: rename VARIABLENAME_PATTERN -> IDENTIFIER
* Parser: intercept errors instead of letting parse tree walker autocorrect and continue.
* New error tests: error_keyword_declare.silly error_keyword_declare2.silly
* Split lowering into two passes, with separate pass for FuncOp, so that we have option of keeping function symbol tables through (dcl/assign/load) op lowering.
* Parser now using symbol table anchored to silly::FuncOp, replacing hashes.  lowering still uses a hash, but it's function:: qualified.
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
* Initial builder support for CALL (fails in lowering.)  Tried using mlir::func::CallOp, but that doesn't like my use of silly::FuncOp instead of mlir::func::FuncOp.  I did that so that my function object had a symbol table for local variables, but it looks like a better approach would be to implement a ScopeOp that has the symbol table, and to then embed ScopeOp in a mlir::func::FuncOp region.
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

The MLIR for the language now matches the statements of the language much more closely.  Consider test.silly for example:

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
  "silly.program"() ({
    "silly.declare"() <{name = "x", type = f64}> : () -> () loc(#loc)
    %0 = "arith.constant"() <{value = 5 : i64}> : () -> i64 loc(#loc1)
    %1 = "arith.constant"() <{value = 3.140000e+00 : f64}> : () -> f64 loc(#loc1)
    %2 = "silly.add"(%0, %1) : (i64, f64) -> f64 loc(#loc1)
    "silly.assign"(%2) <{name = "x"}> : (f64) -> () loc(#loc1)
    %3 = "silly.load"() <{name = "x"}> : () -> f64 loc(#loc2)
    "silly.print"(%3) : (f64) -> () loc(#loc2)
    "silly.declare"() <{name = "y", type = f64}> : () -> () loc(#loc3)
    %4 = "silly.load"() <{name = "x"}> : () -> f64 loc(#loc4)
    %5 = "arith.constant"() <{value = 2 : i64}> : () -> i64 loc(#loc4)
    %6 = "silly.mul"(%4, %5) : (f64, i64) -> f64 loc(#loc4)
    "silly.assign"(%6) <{name = "y"}> : (f64) -> () loc(#loc4)
    %7 = "silly.load"() <{name = "y"}> : () -> f64 loc(#loc5)
    "silly.print"(%7) : (f64) -> () loc(#loc5)
    "silly.exit"() : () -> () loc(#loc)
  }) : () -> () loc(#loc)
}) : () -> () loc(#loc)
#loc = loc("test.silly":1:1)
#loc1 = loc("test.silly":2:5)
#loc2 = loc("test.silly":3:1)
#loc3 = loc("test.silly":4:1)
#loc4 = loc("test.silly":5:5)
#loc5 = loc("test.silly":6:1)
```

I'm still using llvm.alloca, but that now doesn't show up until lowering:

```
; ModuleID = 'test.silly'
source_filename = "test.silly"

declare void @__silly_print(double)

define i32 @main() {
  %1 = alloca double, i64 1, align 8
  store double 8.140000e+00, ptr %1, align 8
  %2 = load double, ptr %1, align 8
  call void @__silly_print(double %2)
  %3 = alloca double, i64 1, align 8
  %4 = load double, ptr %1, align 8
  %5 = fmul double %4, 2.000000e+00
  store double %5, ptr %3, align 8
  %6 = load double, ptr %3, align 8
  call void @__silly_print(double %6)
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
> ./build/silly  samples/foo.silly  -g
"builtin.module"() ({
  "silly.program"() ({
    %0 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<f64> loc(#loc1)
    "silly.declare"() <{name = "x"}> : () -> () loc(#loc1)
    %1 = "arith.constant"() <{value = 3 : i64}> : () -> i64 loc(#loc2)
    %2 = "silly.unary"(%1) <{op = "+"}> : (i64) -> f64 loc(#loc2)
    "memref.store"(%2, %0) : (f64, memref<f64>) -> () loc(#loc2)
    "silly.assign"(%2) <{name = "x"}> : (f64) -> () loc(#loc3)
    "silly.print"(%0) : (memref<f64>) -> () loc(#loc4)
    "silly.return"() : () -> () loc(#loc1)
  }) : () -> () loc(#loc1)
}) : () -> () loc(#loc)
#loc = loc("../samples/foo.silly":1:1)
#loc1 = loc("../samples/foo.silly":1:1)
#loc2 = loc("../samples/foo.silly":2:5)
#loc3 = loc("../samples/foo.silly":2:1)
#loc4 = loc("../samples/foo.silly":4:6)
```

That was removed in V1, which now uses a MLIR dialect that matches the language more closely, deferring alloca to lowering.
