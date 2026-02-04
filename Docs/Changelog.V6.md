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
* test: samples/testerrors.sh -> bin/testerrors

  ci/cd is now effectively:

```bash
  cd samples
  testit
  testerrors
```

* parser:
  * switch to `throw exception_with_context` exclusively for errors (no more asserts other than null pointer checks before dereferences.)
  * buildUnaryExpression.  return the value, instead of pass by reference.
  * buildNonStringUnaryExpression.  New helper function.  Just does the no-string-liternal assertion like check (now throw.)
  * various: add formatLocation( loc ) into the thrown error message where possible.
  * purge auto usage.
  * add asserts before any pointer dereferences
  * convert parser:stripQuotes asserts into throw with context
  * doxygen for parser.hpp
  * make formatLocation const.
  * pass loc down to getFuncOp, getEnclosingScopeOp, and lookupDeclareForVar.
  * getFuncOp/getEnclosingScopeOp: throw if not found.
  * buildUnaryExpression: pass loc as first arg, like most other places.
  * Remove dead code: theTypes, getCompilerType, isBoolean, isInteger, isFloat
  * parser.hpp: move inlines out of class dcl for clarity and put public first.
  * parser.cpp: put all the inlines first.  Uninline a few things.
  * remove a bunch of old if-0'd out `LLVM_DEBUG` code.
  * throw `user_error` for syntax errors, and `exception_with_context` only for internal errors. all the enter/exit callbacks catch `user_error` (all for consistency, even if they don't need to, settting hasErrors and calling mlir::emitError(loc) to bubble the error up to the user.)

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

---

<!-- vim: set tw=80 ts=2 sw=2 et: -->
