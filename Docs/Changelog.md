## tag: V8 (WIP)

* Add a recursion test: factorial.silly
* README: document a libdwarf-tools dependency for dwarfdump (testit)
* PRINT: Allow a list of values, instead of just one (print all to one line.).  Example use case: `factorial.silly`, `print_multiple.silly`.  Implemented in the peeter/old/print-multiple branch, then squashed and merged to master.
* Implemented more complex expressions in parameters.  Example application: factorial.silly: `r = CALL factorial( v - 1 );`  Implemented in the generalized-parameter-value-expressions branch, then squashed and merged to master.  (Have grammar support in place to this in for range variables, but that's not done yet.)
* parser maintainance:
  * rename setFuncOp to setFuncNameAndOp, also passing in the funcName (and have it set currentFuncName)
  * Move the mainScopeGenerated related main() funcOp and scope creation to enterStartRule, removing from getLocations
  * make loc the first param of parseRvalue, like most other functions that take a Location.
  * Doxygen comments for various private functions.
* Allow CALL in unary and binary expressions.
* Expressions are now allowed in for loop range values.  Example program:

```
  INT32 x;
  INT32 a;
  INT32 b;
  INT32 c;
  INT32 z;
  a = 1;
  b = -11;
  c = 2;
  z = 0;

  FOR ( x : (+a, -b, c + z) )
  {
    PRINT x;
  };
```

  Example MLIR fragment for this loop:

```
  %3 = silly.load @a : i32
  %4 = silly.load @b : i32
  %5 = "silly.negate"(%4) : (i32) -> i32
  %6 = silly.load @c : i32
  %7 = silly.load @z : i32
  %8 = "silly.add"(%6, %7) : (i32, i32) -> i32
  scf.for %arg0 = %3 to %5 step %8  : i32 {
    silly.assign @x = %arg0 : i32
    %9 = silly.load @x : i32
    "silly.print"(%9) : (i32) -> ()
  }
```

* Fix functions returning BOOL that have just a RETURN statement.  Eliminates some of the remnants of the old dummy return rewrite code.
* Implement declaration with initialization.
* Add a bunch of grok generated test cases (two of these find errors.)
* Implement return and exit expressions (grammar/parser)
* Add "EXIT 39 + 3" example to README
* Add: exit42.silly
* Add `return_expression.silly` to regression test list, fixing expected output.
* Add error test cases that show that there is no support returning STRING or arrays from functions (grammar only allows scalar types).
* Fix for `minimal_eliftest.silly` duplicate zero output:

  When that was stripped down to just the `check(5)`, I see:

```
  Breakpoint 1, main () at minimal_eliftest.silly:22
  22      CALL check(5);    // should print "positive"
  (gdb) s
  check (val=0) at minimal_eliftest.silly:2
  2       FUNCTION check(INT32 val)
  (gdb) n
  4           IF (val < 0)
  (gdb)
  14              PRINT "zero";
  (gdb)
  zero
  8           ELIF (val > 0)
  (gdb)
  positive
  16          RETURN;
  (gdb)
  main () at minimal_eliftest.silly:2
  2       FUNCTION check(INT32 val)
  (gdb)
  Downloading source file /usr/src/debug/glibc-2.41-11.fc42.aarch64/csu/../sysdeps/nptl/libc_start_call_main.h
  __libc_start_call_main (main=main@entry=0x400634 <_start+52>, argc=argc@entry=1, argv=argv@entry=0xffffffffea28) at ../sysdeps/nptl/libc_start_call_main.h:74
  74        exit (result);
```

  The MLIR was just plain wrong.   There insertion point for the zero PRINT is not in the else block:

```
  func.func private @check(%arg0: i32
    "silly.scope"() ({
      "silly.declare"() <{param_number = 0 : i64, parameter, type = i32}> {sym_name = "val"} : () -> ()
      %0 = silly.load @val : i32
      %c0_i64 = arith.constant 0 : i64
      %1 = "silly.less"(%0, %c0_i64) : (i32, i64) -> i1
      scf.if %1 {
        %2 = "silly.string_literal"() <{value = "negative"}> : () -> !llvm.ptr
        "silly.print"(%2) : (!llvm.ptr) -> ()
      } else {
  >>>>    %2 = "silly.string_literal"() <{value = "zero"}> : () -> !llvm.ptr
  >>>>    "silly.print"(%2) : (!llvm.ptr) -> ()
        %3 = silly.load @val : i32
        %c0_i64_0 = arith.constant 0 : i64
        %4 = "silly.less"(%c0_i64_0, %3) : (i64, i32) -> i1
        scf.if %4 {
          %5 = "silly.string_literal"() <{value = "positive"}> : () -> !llvm.ptr
          "silly.print"(%5) : (!llvm.ptr) -> ()
        } else {
  >>>> should be here.
          }
        }
        "silly.return"() : () -> ()
      }) : () -> ()
      "silly.yield"() : () -> ()
    }
```

 Sure enough the insertion point selection logic was wrong.  The ifOp search logic was finding the outermost scf.if, not the innermost -- now fixed.
* Add a FATAL builtin, like PRINT, but prints any message text, then `FILE:LINE:FATAL error aborting` message and then aborts.  Example program:

```
  INT32 v = 42;

  FATAL "Unexpected value: ", v; // line 3.
```

  Example listing:

```
  module {
    func.func @main() -> i32 {
      "silly.scope"() ({
        "silly.declare"() <{type = i32}> {sym_name = "v"} : () -> () loc(#loc)
        %c42_i64 = arith.constant 42 : i64 loc(#loc)
        silly.assign @v = %c42_i64 : i64 loc(#loc)
        %0 = "silly.string_literal"() <{value = "Unexpected value: "}> : () -> !llvm.ptr loc(#loc1)
        %1 = silly.load @v : i32 loc(#loc1)
        "silly.print"(%0, %1) : (!llvm.ptr, i32) -> () loc(#loc1)
        "silly.abort"() : () -> () loc(#loc1)
        %c0_i32 = arith.constant 0 : i32 loc(#loc)
        "silly.return"(%c0_i32) : (i32) -> () loc(#loc)
      }) : () -> () loc(#loc)
      "silly.yield"() : () -> () loc(#loc2)
    } loc(#loc)
  } loc(#loc)
  #loc = loc("fatal.silly":1:1)
  #loc1 = loc("fatal.silly":3:1)
  #loc2 = loc("fatal.silly":4:1)
```

* Rename FATAL to ERROR, and have that print to stderr, instead of stdout and not abort by itself.  Instead add ABORT statement (no params) that just prints
   the abort message and does so.  Adjusted the lowering for PrintOp and the runtime accordingly.  New sample code:

```
  INT32 v = 42;

  ERROR "Unexpected value: ", v; // line 3.
  ABORT;
```

  MLIR is almost the same, but PRINT now takes an error flag (true in this case):

```
  module {
    func.func @main() -> i32 {
      "silly.scope"() ({
        "silly.declare"() <{type = i32}> {sym_name = "v"} : () -> ()
        %c42_i64 = arith.constant 42 : i64
        silly.assign @v = %c42_i64 : i64
        %0 = "silly.string_literal"() <{value = "Unexpected value: "}> : () -> !llvm.ptr
        %1 = silly.load @v : i32
        %true = arith.constant true
        "silly.print"(%true, %0, %1) : (i1, !llvm.ptr, i32) -> ()
        "silly.abort"() : () -> ()
        %c0_i32 = arith.constant 0 : i32
        "silly.return"(%c0_i32) : (i32) -> ()
      }) : () -> ()
      "silly.yield"() : () -> ()
    }
  }
```

* Adjust test cases to take advantage of new declare w/ initializer syntax
* Add support for array index expressions like t[i+1] or t[CALL someFunc()]
* bin/testit: two new tests, one for index expressions, and another with min/max helper functions (which found a bug.)
* MLIRListener::indexTypeCast: Add support for casting from any size integer type.
* MLIRListener::parsePredicate: Fix bug: was generating silly.less(x,x) instead of (x,y). (t/c: minmax.silly)
* parser: remove parseRvalue std::string argument and push the silly::StringLiteralOp creation logic into there, removing it from processAssignment.
* parser: push the silly::StringLiteralOp logic from parseRvalue down to buildUnaryExpression, and get rid of the last is-string-literal vs. is-not gunk (the StringLiteralOp is now created in buildUnaryExpression and the mlir::Value tested with definingOp.isa instead.)
* PRINT can now take rvalue expressions in the list.  Example: `printexpr.silly`:

```
    STRING s[10] = " there: ";
    INT32 x = 1;
    FLOAT64 f[1];
    f[0] = 3.14;
    FUNCTION foo() : FLOAT32
    {
        RETURN 2.71;
    };

    PRINT "hi", s, 40 + 2, ", ", -x, ", ", f[0], ", ", CALL foo();
```

* Add a CONTINUE parameter to PRINT/ERROR to suppress the newline.
* Implement `--init-fill` option for automatic variable initialization (default zero. test case arrayprod uses 255 -- verified in gdb)
* Removed varStates -- it does not work anyways now that we have control flow.  Still have checking for redeclaration, but no longer have any checking for use without assignment (but have --init-fill to compensate a bit.)
* Reworked the PRINT runtime (and lowering) so that there's now just one function call, with an alloca'ed array (big enough for the largest number of print arguments in the function in question.)  For example, given this MLIR fragment:

```
  %c42_i64 = arith.constant 42 : i64
  %c0_i32 = arith.constant 0 : i32
  "silly.print"(%c0_i32, %c42_i64) : (i32, i64) -> ()

  %cst = arith.constant 4.200000e+01 : f64
  %c0_i32_0 = arith.constant 0 : i32
  "silly.print"(%c0_i32_0, %cst) : (i32, f64) -> ()

  %c1_i64 = arith.constant 1 : i64
  %c2_i64 = arith.constant 2 : i64
  %c0_i32_1 = arith.constant 0 : i32
  "silly.print"(%c0_i32_1, %c1_i64, %c2_i64) : (i32, i64, i64) -> ()
```

  lowered to:

```
  %1 = alloca [4 x { i32, i32, i64, ptr }], align 8
  store i32 1, ptr %1, align 8
  %.repack1 = getelementptr inbounds nuw i8, ptr %1, i64 4
  store i32 0, ptr %.repack1, align 4
  %.repack2 = getelementptr inbounds nuw i8, ptr %1, i64 8
  store i64 42, ptr %.repack2, align 8
  %.repack3 = getelementptr inbounds nuw i8, ptr %1, i64 16
  store ptr null, ptr %.repack3, align 8
  call void @__silly_print(i32 1, ptr nonnull %1), !dbg !8

  store i32 2, ptr %1, align 8
  store i32 0, ptr %.repack1, align 4
  store i64 4631107791820423168, ptr %.repack2, align 8
  store ptr null, ptr %.repack3, align 8
  call void @__silly_print(i32 1, ptr nonnull %1), !dbg !9

  store i32 1, ptr %1, align 8
  store i32 1, ptr %.repack1, align 4
  store i64 1, ptr %.repack2, align 8
  store ptr null, ptr %.repack3, align 8
  %2 = getelementptr inbounds nuw i8, ptr %1, i64 24
  store i32 1, ptr %2, align 8
  %.repack10 = getelementptr inbounds nuw i8, ptr %1, i64 28
  store i32 0, ptr %.repack10, align 4
  %.repack11 = getelementptr inbounds nuw i8, ptr %1, i64 32
  store i64 2, ptr %.repack11, align 8
  %.repack12 = getelementptr inbounds nuw i8, ptr %1, i64 40
  store ptr null, ptr %.repack12, align 8
  call void @__silly_print(i32 2, ptr nonnull %1), !dbg !10
```

* new helper functions in unary expression parsing: parseBoolean, parseInteger, parseFloat (also use for initializer list element parsing/building.)
* grammar: Remove `INTEGER_PATTERN` from booleanLiteral.  Didn't have any test that relied on or tested that codepath.
* Now have grammar/builder support for initializer lists.  Example MLIR:

```
  "silly.declare"() <{type = i1}> {sym_name = "j"} : () -> ()
  "silly.declare"(%true) <{type = i1}> {sym_name = "i"} : (i1) -> ()
  "silly.declare"(%false) <{type = i1}> {sym_name = "h"} : (i1) -> ()
  "silly.declare"(%c1_i64) <{type = f64}> {sym_name = "g"} : (i64) -> ()
  "silly.declare"(%cst, %cst_0) <{size = 3 : i64, type = f32}> {sym_name = "f"} : (f64, f64) -> ()
  "silly.declare"(%c1_i64_1, %c2_i64, %c3_i64) <{size = 3 : i64, type = i32}> {sym_name = "e"} : (i64, i64, i64) -> ()
  "silly.declare"(%c1_i64_2) <{type = i64}> {sym_name = "d"} : (i64) -> ()
  "silly.declare"(%c1_i64_3, %c2_i64_4) <{size = 3 : i64, type = i32}> {sym_name = "c"} : (i64, i64) -> ()
  "silly.declare"(%c1_i64_5) <{type = i16}> {sym_name = "b"} : (i64) -> ()
  "silly.declare"(%c1_i64_6) <{type = i8}> {sym_name = "a"} : (i64) -> ()
```

* Adjust bitwiseop.perl compareop.perl to use the new multi-argument PRINT support in all the generated tests.
* initializer-syntax now works all the way from grammar to lowering.  Documented it in the README.
* Initialization syntax implemented for strings too.  test case: initstring.silly
* (branch peeter/old/complex-expressions -- squashed and cherry-picked to master).  Implemented more complex expressions (chains of operators...).  Examples:

```
  DCL x;
  x = 5 + 3 * 2;
  PRINT x;
  PRINT 1 + 2 * (3 - 4);  // -1
  PRINT - - 5;            // 5
  PRINT NOT TRUE;         // 0 (FALSE)
  PRINT (10 < 20) EQ 1;   // 1 (TRUE)
```

* Prohibit NOT on non-integer type.  t/c: `error_notfloat.silly`
* [grammar/parser] Prohibit chaining of Comparison operators (examples: `1 < 2 < 3', `1 EQ 1 NE 1`).
* [grammar/parser] Introduce operator grammar elements (example: multiplicativeOperator) so that I don't have ambiguous arrays of terminal-nodes.  This way I get an array of multiplicativeOperator, each with it's own terminal node.  This fixes samples/expression5.silly, now enabled in the regression suite.

* [grammar, parser] Add Statement suffixes to a bunch of the rules (abort, ifelifelse, for, print, assignment, declare, intDeclare, floatDeclare, stringDeclare, boolDeclare, function, get, error)
* Fixed previously documented, but stupid, semantics:

 Computations occur in assignment operations, and any types are first promoted to the type of the variable.
 This means that `x = 1.99 + 2.99` has the value `3`, if `x` is an integer variable, but `4.98` if x is a `FLOAT32` or `FLOAT64`.
* Document: Floating point to integer conversions use a floor operation, and are not rounded.
* A bunch more tests for complex expressions and more.
* Generalize initializer expressions, and fix +- in unaryExpression ambiguity.

  * [grammar, parser] Allow parameter+constant expressions
  * [parser] Remove the previous insertion point hack for declarations.  They are now in program order again (but still hoisted up to the begining of the scope.)
  * [grammar, parser] remove booleanValue, replacing with direct use of expression Grammar/parser now compiles, but the insertion point logic is broken: My declaration order hack is now causing trouble.
  * [test] add declaration order test case.
  * [grammar, parser] add declareAssignmentExpression, and remove (PLUSCHAR_TOKEN | MINUS_TOKEN)? from the INTEGER/FLOAT literal patterns.
  * Enable new tests: array_dynamic_index, array_in_expr, array_in_expr_min.
  * [grammar, parser] remove booleanValue, replacing with direct use of expression
  * [grammar, parser] Replace assignmentRvalue with expression in assignmentStatement.  Use declareAssignmentExpression as expression alias, but just in the declaration rules (to distinguish assignment from initialization.)

  Details:

  The tests array_dynamic_index, array_in_expr, array_in_expr_min, as well
  as testing more general array element expressions, found a fundamental
  grammar bug.

  A minimal reproducer was:

```
  INT32 i;
  i = 2+1; // parse fails.
  //i = 2 + 1; // parse okay.
```

  The root cause of this was the `(PLUSCHAR_TOKEN | MINUS_TOKEN)?` that was in `FLOAT_PATTERN` and `INTEGER_PATTERN`.
  This leads to parse ambiguity since the +- in unaryExpression and this ended up in conflict.

  Fixing this wasn't as easy as just removing those patterns, letting unaryExpression take the load.
  Trying that broke initializer-lists which only allowed literal expressions.

  By first allowing expressions in initializer-list elements (requiring
  extensive parser changes), it was then possible to remove `(PLUSCHAR_TOKEN | MINUS_TOKEN)?` from the
  FLOAT and INTEGER patterns.

 
* more tests: `init_expr_unary initlisttrunc initsequence init_expr_bool forward_ref_init init_expr_call`
* add placeholder test: `error_nonconst_init.silly` -- doesn't fail, but I want it to (see TODO)
* `perl -p -i -e 's/MLIRListener/ParseListener/g'` -- Bad name: It's an MLIR builder or Silly Parse Listener, but not an MLIRListener
* Add new error message: return expression found '{}', but no return type for function {}
* Add corresponding test case: error_return_expr_no_return_type.silly
* Fix array_lvalue_complex.silly (user error -- above).
* Fix initlist_param.silly -- that was a test to see that an initializer-list expression can reference a parameter (it still shouldn't reference a variable with the current implementation).  That now works.  The issue was the DeclareOp sequencing for the parameters vs. the variables -- createScope now saves the last declareOp creation point, like registerDeclaration does.  Removes the setFuncNameAndOp() helper function so the consistuient parts of that function can be split up.
* Fix the location information for expressions.  location info is now granular, so an expression like c[i] will have a location for i and one for the i.  Example (line 9):

```
  PRINT "c[", i, "] = ", t;
```

  The MLIR for that is:

```
  %7 = "silly.string_literal"() <{value = "c["}> : () -> !llvm.ptr loc(#loc14)
  %8 = silly.load @i : i32 loc(#loc15)
  %9 = "silly.string_literal"() <{value = "] = "}> : () -> !llvm.ptr loc(#loc16)
  %10 = silly.load @t : i32 loc(#loc17)
  %c0_i32_2 = arith.constant 0 : i32 loc(#loc18)
  "silly.print"(%c0_i32_2, %7, %8, %9, %10) : (i32, !llvm.ptr, i32, !llvm.ptr, i32) -> () loc(#loc18)
  ...

  #loc14 = loc("printdi.silly":9:11)
  #loc15 = loc("printdi.silly":9:17)
  #loc16 = loc("printdi.silly":9:20)
  #loc17 = loc("printdi.silly":9:28)
  #loc18 = loc("printdi.silly":9:5)
```

* Switch scf.for loop bodies to use a proper SSA form.
  * [grammar] Introduce intType rule (use in intDeclareStatement and forStatement)
  * [parser] Add vector<pair<string, Value>> for induction variables and push/pop that in the FOR loop enter/exit callbacks.  Split out integerDeclarationType from enterIntDeclareStatement to also use in enterFor.
  * [parser] ParseListener::parsePrimary -- supplement variable lookup with induction var lookup.
  * Example of the new MLIR for a loop:

```
    scf.for %arg0 = %0 to %1 step %2  : i32 {
      %3 = arith.extsi %arg0 : i32 to i64 loc(#loc10)
      %4 = arith.index_cast %3 : i64 to index loc(#loc10)
      %5 = silly.load @c[%4] : i32 loc(#loc11)
      silly.assign @t = %5 : i32 loc(#loc12)
      %6 = "silly.string_literal"() <{value = "c["}> : () -> !llvm.ptr loc(#loc13)
      %7 = "silly.string_literal"() <{value = "] = "}> : () -> !llvm.ptr loc(#loc14)
      %8 = silly.load @t : i32 loc(#loc15)
      %c0_i32_2 = arith.constant 0 : i32 loc(#loc16)
      "silly.print"(%c0_i32_2, %6, %arg0, %7, %8) : (i32, !llvm.ptr, i32, !llvm.ptr, i32) -> () loc(#loc16)
    } loc(#loc9)
```

  This removes the AssignOp for the loop induction variable that I used to avoid figuring out how to cache and
  lookup the mlir::Value for the induction var.  Unfortunately, this means that I loose the debug instrumentation
  for that loop variable as a side effect.  Also unfortunately, this also doesn't fix the line number ping pong
  that I am seeing in loop bodies.  More debugging of the DI is required.

  * More FOR tests. Also implemented checking of the error messages for expected compile errors.
  * [parser] searchForInduction doesn't need to return a pair -- only the Value is ever used.
  * [parser] pass type from parseExpression all the way down to parsePrimary, and adjust constant creation to use that type if specified.  This gets rid of a bunch of ugly sign-extension/truncation.  Example:

```
    %0 = "arith.constant"() <{value = 0 : i32}> : () -> i32 loc(#loc2)
    %1 = "arith.constant"() <{value = 3 : i32}> : () -> i32 loc(#loc3)
    %2 = "arith.constant"() <{value = 1 : i32}> : () -> i32 loc(#loc1)
    "scf.for"(%0, %1, %2) ({
    ^bb0(%arg0: i32 loc("nested_for.silly":5:1)):
      %3 = "arith.constant"() <{value = 0 : i32}> : () -> i32 loc(#loc4)
      %4 = "arith.constant"() <{value = 2 : i32}> : () -> i32 loc(#loc5)
      %5 = "arith.constant"() <{value = 1 : i32}> : () -> i32 loc(#loc6)
      "scf.for"(%3, %4, %5) ({
```

 With all the previous truncatation (from i64) this was really hard to read.
* [parser] pass the type into the initializer-list creation so that no casting will be required in lowering.  Example:

```
  INT32 c[3]{10,20,30};
```

  gives

```
  %0 = "arith.constant"() <{value = 10 : i32}> : () -> i32 loc(#loc2)
  %1 = "arith.constant"() <{value = 20 : i32}> : () -> i32 loc(#loc3)
  %2 = "arith.constant"() <{value = 30 : i32}> : () -> i32 loc(#loc4)
  "silly.declare"(%0, %1, %2) <{size = 3 : i64, type = i32}> {sym_name = "c"} : (i32, i32, i32) -> () loc(#loc5)
```

(everything here has the right type now (not i64), right out of the gate.)
* Fix nested_for.silly.  Handle insertionPointStack like lastDeclareOp, pushing getOperation() and restoring using  builder.setInsertionPointAfter.

  Was seeing:

```
  "scf.for"(%3, %4, %5) ({
    ^bb0(%arg0: i32 loc("nested_for.silly":5:1)):
      %6 = "arith.constant"() <{value = 0 : i32}> : () -> i32 loc(#loc9)
      %7 = "arith.constant"() <{value = 2 : i32}> : () -> i32 loc(#loc10)
      %8 = "arith.constant"() <{value = 1 : i32}> : () -> i32 loc(#loc11)
      "scf.for"(%6, %7, %8) ({
      ^bb0(%arg1: i32 loc("nested_for.silly":6:5)):
      ...
        "scf.yield"() : () -> () loc(#loc11)
      }) : (i32, i32, i32) -> () loc(#loc11)
      %9 = "arith.constant"() <{value = 0 : i32}> : () -> i32 loc(#loc1)
      "silly.return"(%9) : (i32) -> () loc(#loc1)
      "scf.yield"() : () -> () loc(#loc8)
    }) : (i32, i32, i32) -> () loc(#loc8)
  }) : () -> () loc(#loc1)

  RETURN SHOULD BE HERE ... it's up in the outer for loop body!

    "silly.yield"() : () -> () loc(#loc20)
  }) : () -> () loc(#loc1)
```

* Document that negative and zero size step values in FOR loops is not supported, and has undefined behaviour.
* Implement FOR induction variable debug-instrumentation.
  * [tablegen] Add silly::debug_name
  * [parser] resurrect getTerminalLocation, and use it to construct a DebugName OP in enterFor.
  * [lowering] split out infoForVariableDI from constructVariableDI for use in DebugNameOpLowering.
    Implement DebugNameOpLowering (mostly calling new helper constructInductionVariableDI), and make DebugName an illegalop.

  Example.  MLIR fragment:

```
  "silly.debug_name"(%arg0) <{name = "i"}> : (i64) -> () loc(#loc4)
  loc4 = loc("for_simplest.silly":3:12)
```

  Example LLVM-IR fragment:

```
  %lsr.iv = phi i64 [ %lsr.iv.next, %6 ], [ 2, %0 ], !dbg !11
  ...
  #dbg_value(i64 %lsr.iv, !12, !DIExpression(DW_OP_constu, 1, DW_OP_minus, DW_OP_stack_value), !13)

  !12 = !DILocalVariable(name: "i", scope: !4, file: !1, line: 3, type: !10, align: 64)
```

  Example debug session:

```
  (gdb) run
  Starting program: /home/peeter/toycalculator/samples/out/for_simplest
  Downloading separate debug info for system-supplied DSO at 0xfffff7ffa000
  [Thread debugging using libthread_db enabled]
  Using host libthread_db library "/lib64/libthread_db.so.1".

  Breakpoint 2, main () at for_simplest.silly:5
  5           PRINT i;
  (gdb) n
  1
  6           v = i + 1;
  (gdb) p i
  $1 = 1
  (gdb) n
  3       FOR (INT64 i : (1, 5))
  (gdb) n

  Breakpoint 2, main () at for_simplest.silly:5
  5           PRINT i;
  (gdb) p i
  $2 = 2
  (gdb) what i
  type = int64_t
```

* [lowering] Standardize on a 'loc, rewriter' sequence in all helper functions that take both (instead of a hodge podge or 'loc, rewriter' and 'rewriter, loc')
* Very simplest debug test case for loop induction variables.  For test `for_simplest` check the dwarfdump for:

```
  DW_AT_name                  i
  DW_AT_alignment             0x00000008
  DW_AT_decl_file             0x00000001 ./for_simplest.silly
  DW_AT_decl_line             0x00000003
```

* loadstore.silly -- This contains all the most basic load and store accesses:
  * Scalar load and store
  * Array element access: load and store
  * string store and load and string literal access.

  This program was specifically for examining the MLIR silly dialect representation of those operations.

* Disable: `div_zero_int` -- different results on intel vs. arm.
* Refactor DeclareOp, AssignOp and LoadOp to use a proper SSA form (from branch peeter/ssa-form-assign-load, squashed and cherry-picked into master.)

  * [lowering] Adjust all the DeclareOp, AssignOp and LoadOp's to use the new model.
  * [parser] Adjust all the DeclareOp, AssignOp and LoadOp's to use the new model.
  * [cmake] Generate SillyTypes.hpp.inc SillyTypes.cpp.inc, adding dependencies.
  * [TODO] No error for "Attempted GET to string literal"
  * [TODO] nice to have: custom var printer to show scalar type, as `<i64>` (for example) instead of `<i64 []>`
  * [tablegen] New Silly_VarType, use in DeclareOp, returning that varType (still has the var_name symbol.)  Adjusted AssignOp and LoadOp to use %foo (a DeclareOp mlir::Value) instead of a var_name symbol reference.  Adjusted all the source/headers that include the tablegen boilerplate headers -- lots of tweaking required.

  Details:

  This introduces a new Silly_VarType, now used in DeclareOp.  AssignOp and LoadOp's now refer to a DeclareOp, instead of a var_name (symbol reference)

  The varType:

```
  def Silly_VarType : TypeDef<Silly_Dialect, "var"> {
    let summary = "Abstract variable location (scalar or array)";
    let description = [{
      Represents an abstract handle to a variable's storage.
      Scalars have empty shape; arrays have a non-empty shape.
      This type is storage-agnostic and lowered to concrete memory
      (e.g., LLVM alloca).
    }];

    let parameters = (ins
      "mlir::Type":$elementType,
      "mlir::DenseI64ArrayAttr":$shape // empty = scalar
    );

    let mnemonic = "var";

    let assemblyFormat = "`<` $elementType $shape `>`";
  }
```

  Declare's now look like:

```
  %0 = silly.declare %c1_i64 : i64 {sym_name = "anInitializedScalar"} : <i64 []>
  %1 = silly.declare  :  {sym_name = "aTemporaryScalar"} : <i64 []>
  %2 = silly.declare %c1_i64_0, %c0_i64 : i64, i64 {sym_name = "aVectorLength2"} : <i64 [2]>
  %3 = silly.declare  :  {sym_name = "aStringLength2"} : <i8 [10]>
```

  Example LoadOp, and AssignOp's

```
  %1 = silly.declare  :  {sym_name = "aTemporaryScalar"} : <i64 []>
  %c3_i64 = arith.constant 3 : i64
  silly.assign %1 : <i64 []> = %c3_i64 : i64

  %4 = silly.load %1 : <i64 []> : i64
  ...
  "silly.print"(%c0_i32, %4) : (i32, i64) -> ()
```

  Loads and Assigns now reference a DeclareOp (SSA return value), and not a symbol name.

* [types] custom var printer to show scalar type, as `<i64>` (for example) instead of `<i64 []>`.  Unfortunately that required a parse method too.
* [cmake] package silly dialect files in libSillyDialect.so instead of static linking to the silly compiler driver.  This should at least theoretically allow the use of mlir-opt to test parse directly (not tried yet.)  Also comment out the mlirtest and simplest test build rules -- I haven't tried those in forever, and don't care to at the moment.
* [tablegen] remove the DeclareOp custom asm printer.  It generates output that can't be parsed by mlir-opt.
* [dialect] convert library into a plugin so it can be loaded by mlir-opt.
* [tests] Add manual tests tests/dialect/.
* [bin] Add: silly-opt -- an easy way to run mlir-opt against a silly dialect file.
* Add lit test build infrastructure -- unfortunately depends on my llvm-build/ dir and llvm-project cmake configuration -- but it's a CI/CD start.
* [tests] ReturnOp verifier and some initial returnop dialect tests.  Move the {ScopeOp,DeclareOp}::verify out of line.
* [tests] Migrated testit manual testsuite driver to ctest.
* [tests] Add testit --clean option.  default off, so that ctest -j works.
* Replace lowering asserts like:

```
  assert( false && "unsupported print argument type" );
```

  With:

```
  return rewriter.notifyMatchFailure( op, "unsupported print argument type" );
```

  * change return type of helpers to mlir::LogicalResult (passing outputs by reference if required, and op's as input), then:
  * Add idiomatic MLIR checking of the form:

```
    if ( mlir::failed( lState.createGetCall( loc, rewriter, op, inputType, result ) ) )
    {
        return mlir::failure();
    }
```

* [README] start making the intro more comprehensive.
* [lowering] replace the last throw with notifyMatchFailure.
* [docs] move Changelog.md `Claude_code_review_Feb_2026.md` TODO.md to Docs/
* [build] Introduce some directory hierarchy.  Build artifacts are now:

```
  build/bin/silly
  build/lib/libsilly_runtime.so
  build/lib/libSillyDialect.so
```

  with sources in the following directories:

```
src/runtime/
src/dialect/
src/driver/
src/include/
src/grammar/
```

  Each of these directories (except src/include/) now has it's own CMakeLists.txt, so you can look at the build rules for each component in isolation.  The top level makefile is now pretty minimal.
* [tests] Move samples/ to tests/endtoend/, and parsetests/ to tests/dialect/ (adjusting cmake and scripts and docs accordingly)
* [layout] move some old crud out of bin/ (to bin/.old for now.)
* [tests] Fix: testsuite leaves crap in: tests/dialect/tests/dialect/ -- it's in the gitignore, but would be better in build/
* [grammar,parser,readme,tests] Get rid of DCL/DECLARE
* [make] /build/ no longer hardcoded in bin/silly-opt and bin/testit
* [tests] introduce a directory heirarchy for the endtoend tests.
* [tests] fill in some of the verify() testing coverage holes.
