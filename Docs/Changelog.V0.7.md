## tag: V7 (Jan 4, 2025)

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

* PRINT: Also implement BOOLean print literal (t/c: printboollit.silly)

  MLIR:

```
  module {
    func.func @main() -> i32 {
      "silly.scope"() ({
        %true = arith.constant true
        silly.print %true : i1
        %false = arith.constant false
        silly.print %false : i1
        %c0_i32 = arith.constant 0 : i32
        "silly.return"(%c0_i32) : (i32) -> ()
      }) : () -> ()
      "silly.yield"() : () -> ()
    }
  }
```

  LL:

```
  declare void @__silly_print_i64(i64)

  define i32 @main() !dbg !4 {
    call void @__silly_print_i64(i64 1), !dbg !8
    call void @__silly_print_i64(i64 0), !dbg !9
    ret i32 0, !dbg !9
  }
```

### 2. README now has a language-reference.

### 3. Implemented ELIF, plus parser location logic simplification:

* parser: merge mainFirstTime into getLocation, removing all calls to it.  i.e.: any time we process a location for the first time is a good time to generate the main wrapper.
* note bug in function.silly DI (while stepping into CALL, not just after.)
* make getLocation bool param required and explicit.
* purge parser functions: setLastLoc, getLastLoc (and PerFuncState::lastLoc).  Switched to getStop() based token location.
* Gut the dummy return-logic in createScope, and generate return/exit at function/program exit instead.  This allows for purging the terminator field in PerFunctionState too.
* Fix the DISubprogramAttr line and scopeline parameters (handles stepping into CALL getting the line numbers wrong).  Simple test case added.
* driver: Fix deprecated overload warning in TargetMachine construction
* not.silly: more comprehensive all types testing for unary NOT operator.
* lowering: purge the last auto variables
* Implement ELIF.  Was done in the peeter/old/elif-support branch, squashed when merged to master.  This simplifies the IF/ELSE handling considerably.  In particular, I now generate scf.if with both then/else regions automatically, so that there's an implicit terminator.  This means that a program like the following:

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

  now shows up with an empty else block in the MLIR representation:

```
  module {
    func.func @main() -> i32 {
      "silly.scope"() ({
        "silly.declare"() <{type = i32}> {sym_name = "y"} : () -> ()
        "silly.declare"() <{type = i32}> {sym_name = "x"} : () -> ()
        %c3_i64 = arith.constant 3 : i64
        silly.assign @x = %c3_i64 : i64
        %0 = silly.load @x : i32
        %c4_i64 = arith.constant 4 : i64
        %1 = "silly.less"(%0, %c4_i64) : (i32, i64) -> i1
        scf.if %1 {
          %c42_i64 = arith.constant 42 : i64
          silly.assign @y = %c42_i64 : i64
          %3 = silly.load @y : i32
          silly.print %3 : i32
        } else {
        }
        %2 = "silly.string_literal"() <{value = "Done."}> : () -> !llvm.ptr
        silly.print %2 : !llvm.ptr
        %c0_i32 = arith.constant 0 : i32
        "silly.return"(%c0_i32) : (i32) -> ()
      }) : () -> ()
      "silly.yield"() : () -> ()
    }
  }
```

  We can represent a program with an ELIF fairly easily as a nested scf.if in that new else block:

```
  INT32 x; // line 1

  x = 3; // line 3

  IF ( x < 4 ) // line 5
  {
     PRINT x; // line 7
  }
  ELIF ( x > 5 ) // line 9
  {
     PRINT "Bug if we get here."; // line 11
  };

  PRINT 42; // line 13
```

  This looks like:

```
  module {
    func.func @main() -> i32 {
      "silly.scope"() ({
        "silly.declare"() <{type = i32}> {sym_name = "x"} : () -> () loc(#loc)
        %c3_i64 = arith.constant 3 : i64 loc(#loc1)
        silly.assign @x = %c3_i64 : i64 loc(#loc1)
        %0 = silly.load @x : i32 loc(#loc2)
        %c4_i64 = arith.constant 4 : i64 loc(#loc2)
        %1 = "silly.less"(%0, %c4_i64) : (i32, i64) -> i1 loc(#loc2)
        scf.if %1 {
          %2 = silly.load @x : i32 loc(#loc3)
          silly.print %2 : i32 loc(#loc3)
        } else {
          %2 = silly.load @x : i32 loc(#loc4)
          %c5_i64 = arith.constant 5 : i64 loc(#loc4)
          %3 = "silly.less"(%c5_i64, %2) : (i64, i32) -> i1 loc(#loc4)
          scf.if %3 {
            %4 = "silly.string_literal"() <{value = "Bug if we get here."}> : () -> !llvm.ptr loc(#loc5)
            silly.print %4 : !llvm.ptr loc(#loc5)
          } else {
          } loc(#loc4)
        } loc(#loc2)
        %c42_i64 = arith.constant 42 : i64 loc(#loc6)
        silly.print %c42_i64 : i64 loc(#loc6)
        %c0_i32 = arith.constant 0 : i32 loc(#loc)
        "silly.return"(%c0_i32) : (i32) -> () loc(#loc)
      }) : () -> () loc(#loc)
      "silly.yield"() : () -> () loc(#loc7)
    } loc(#loc)
  } loc(#loc)
  #loc = loc("elif.silly":1:1)
  #loc1 = loc("elif.silly":3:5)
  #loc2 = loc("elif.silly":5:1)
  #loc3 = loc("elif.silly":7:4)
  #loc4 = loc("elif.silly":9:1)
  #loc5 = loc("elif.silly":11:4)
  #loc6 = loc("elif.silly":14:1)
  #loc7 = loc("elif.silly":1:7)
```

---

<!-- vim: set tw=80 ts=2 sw=2 et: -->
