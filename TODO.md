## TODO

----------------------------------
1. array index expressions aren't working: `array_in_expr_min, array_in_expr, array_lvalue_complex`

  See notes in d967fba53747e337c70c7cbd356f582c8c0100e0, and experiment hack in `experiment-with-fixing-array_in_expr_min` (on fedoravm, not pushed.)

  Think that the proper fix for this is to:
  - Allow `expression` in initializer-list (and declare w/ assignment?)
  - Remove `(PLUSCHAR_TOKEN | MINUS_TOKEN)?` from both `FLOAT_PATTERN` and `INTEGER_PATTERN`, so that unaryExpression is the sole source of +-

2. allow: INT64 a = 1, b = 2, c = 3; (`chained_comparison_parens.silly`).
3. forgetting RETURN in `array_elem_as_arg.silly` has a very confusing error.
4. grammar probably allows for function declared in a function.  prohibit that or at least test for it?

----------------------------------
* Bugs:
  - `error_intarray_bad_constaccess.silly` should fail but doesn't.
  - Have ABORT builtin now to implement failure codepath for runtime array range checking.  Could also use this for assignment to bool from source value > 1.
  - line stepping over a multi-argument print statement is getting a new line entry for each individual runtime call:

```
  (gdb) p w
$3 = 0
(gdb) n
34          PRINT "c[", i, "] = ", c[i], " ( = ", v, " * ", w, " )";
(gdb)
32          c[i] = v * w;
(gdb)
34          PRINT "c[", i, "] = ", c[i], " ( = ", v, " * ", w, " )";
(gdb)
32          c[i] = v * w;
(gdb)
34          PRINT "c[", i, "] = ", c[i], " ( = ", v, " * ", w, " )";
(gdb)
32          c[i] = v * w;
(gdb)
34          PRINT "c[", i, "] = ", c[i], " ( = ", v, " * ", w, " )";
(gdb)
c[0] = 0 ( = 1 * 0 )
14      FOR (i: (0, 10))
```

  I suppressed real location info for all but the last of these PRINT calls, but that didn't handle it completely:
```
(gdb)
32          c[i] = v * w;
(gdb)

Breakpoint 1, main () at arrayprod.silly:34
34          PRINT "c[", i, "] = ", c[i], " ( = ", v, " * ", w, " )";
(gdb)
32          c[i] = v * w;
(gdb)
34          PRINT "c[", i, "] = ", c[i], " ( = ", v, " * ", w, " )";
(gdb)
c[1] = 0 ( = 2 * 0 )
14      FOR (i: (0, 10))
```

  Grok suggests suppressing all real location information for the geps and other variable references that are probably bouncing the lines around (or reworking the runtime so there isn't multiple print function calls.)  I've since eliminated the multiple print calls,
  but still have this issue in arrayprod.  Using unknownLoc for the print args doesn't help -- tried as an experiment.

  ```
  %8 = bitcast ptr %1 to ptr, !dbg !24
  store { i32, i32, i64, ptr } { i32 1, i32 0, i64 42, ptr null }, ptr %8, align 8, !dbg !24
  call void @__silly_print(i32 1, ptr %1), !dbg !24
  ```

  - In initFill: `loc = rewriter.getUnknownLoc();` This was a HACK to suppress location info for these implicit memset's, so that the line numbers in gdb don't bounce around.  I think that the re-ordering that I now do in the DeclareOp builder is messing things up, and that is the root cause of the trouble (see that in arrayprod.silly trying to step over the dcls if this hack is removed.)

----------------------------------
* Test:

  - More testing for FOR including (dcl, assignment, call, IF/ELIF/ELSE, ...).
  - Tests for all the type conversions (i.e.: binary and unary arith operators)
  - More tests for array access and assignment.  Example, some vector expressions (add, convolution, ...).
  - Tests for call in binary and unary expressions.
  - Have AI tool review existing tests, looking for holes relative to the grammar.  Started this... revisit for error test cases.
    - same for the README "language reference"

----------------------------------
* Expressions:
  - Assignment like initialization syntax is not available for array variables.  That should trigger an error.
  - Maybe: Get rid of CALL and support calls in binary expressions: r = v * foo().
  - Expressions that aren't parsed properly (like `CALL factorial(v - 1)` used to) lead to mysterious seeming parse error: Should do better.
  - declaration scope is weird, persisting beyond the declaring block (see: scopebug.silly and the README)
  - NOT should be allowed for boolean expressions in IF/ELIF predicates.
  - no test case for CALL in booleanExpression
  - grammar allows for CALL to have CALL in the parameter list.  Not tested.
  - PRINT now supports rvalue-expressions.  I had to hack it to interpret INT8 [] arrays as strings in parseRvalue (for PRINT/ERROR handling cases only), as strings aren't a first class entity -- fix that.

----------------------------------
* Maintainance:

  - All the runtime functions should take location context to show where in the users code the error was, if one happens (i.e.: GET functions)
  - lowering error handling is pschizophrenic: mix of: assert(), throw, llvm::unreachable, rewriter.notifyMatchFailure, emitError, ...
  - Move scf lowering into 1st pass?  Attempt at this in xpgn:streamline-passes-attempt branch (not pushed.)
  - Lots of cut and paste duplication for type conversion in lowering.cpp -- split out into helper functions.
  - Final version of ELIF support meant that I didn't need getStopLocation() anymore (was using it to generate scf.yield with the end of the block location).  Review that and decide what to do with it.
  - merge these?

```
         registerDeclaration( loc, varName, tyF64, ctx->arrayBoundsExpression() );

         if ( SillyParser::AssignmentRvalueContext *expr = ctx->assignmentRvalue() )
         {
             processAssignment( loc, expr->rvalueExpression(), varName, {} );
         }
```

----------------------------------

* Exit/return
  - Maybe: enforce i8 Exit type in the grammar (i.e.: actual UNIX shell semantics.)  Don't have an ExitOp anymore, just ReturnOp, but could switch from the hardcoded i32 return type in the builder.
  - Allow EXIT at more than the end of program (that restriction is currently enforced in the grammar.)
  - allow return/exit at other places in a function/program so get out prematurely.
  - remove the mandatory RETURN in the FUNCTION grammar, and mirror the exitHandled

----------------------------------

* String tests and behaviour:
  - longstring.silly:
   compiler truncates an assignment from a too long string literal.  Could implement safe vs. checked behaviour so that such an assignment would trigger an error instead.

----------------------------------
* Control flow:
  - Should implement a break-like keyword for the FOR loop.  That would allow for a "poor man's while", with an effectively infinite very large loop bound.
  - AND/OR: Implement WHILE/DO/BREAK/CONTINUE statements.

----------------------------------
* Tablegen/MLIR:
  - Don't have any traits defined for my MLIR operations (initially caused compile errors, and I just commented-out or omitted after that.)
  - Validators, and better custom assembly printers.
  - MLIR has considerable capability for semantic checking, but I'm not exploiting that here, and have very little in custom verifiers.
  - If verify fails, I shouldn't proceed to lowering.  Do I?

----------------------------------
* arrays:
  - implement runtime bounds checking (make it a compiler option?)
  - attempt to put in constant array access range checking did not work (for AssignOp lowering, probably also for LoadOp lowering -- untested).

----------------------------------
* Misc:
  - GET into a BOOL should logically support TRUE/FALSE values, and not just 0/1.  PRINT of a BOOL should display TRUE/FALSE, not just 1/0.
  - Write a MLIR walker (and/or opt-silly front end) to see how to answer code questions about a given program.

  - More complicated expressions.
  - CAST operators.  Could also implement that with a "standard library", or on demand.

  - Just for fun: Implement a JIT so that the "language" has an interpretor mode, as well as static compilation.

----------------------------------
* Debugging:

  - Expand Dwarf DI test cases (have just one in `bin/testit` for `samples/f.silly`, and it is completely manual, and only checks bar0 line number.)

  - t/c: function.silly: see if this is still and issue, and debug it, if it is:

    line stepping behaves wrong after CALL too, showing the wrong line after some calls:

```
        (gdb) n
        main () at function.silly:38
        38      v = CALL add( 42, 0 );
        (gdb) s
        add (v=0, w=4196336) at function.silly:13
        13      FUNCTION add( INT32 v, INT32 w ) : INT32
        (gdb) n
        16          r = v + w;
        (gdb) n
        17          RETURN r;
        (gdb) p r
        $1 = 42
        (gdb) p v
        $2 = 42
        (gdb) p w
        $3 = 0
        (gdb) n
        main () at function.silly:39
        39      PRINT v;                    << OKAY
        (gdb) n
        38      v = CALL add( 42, 0 );      << WRONG, ALREADY DONE.
        (gdb) s
        39      PRINT v;                    << WRONG, ALREADY DONE.
        (gdb) n
        42
        40      v = CALL zero();
```

    sequence should have been:

```
     37 INT32 v;
     38 v = CALL add( 42, 0 );
     39 PRINT v;
     40 v = CALL zero();
     41 PRINT v;
     42 v = CALL plus3( 39 );
     43 PRINT v;
```
  - gdb session for simpleless.silly is not behaving right with respect to 'next'.  Suspect that this is due to my cachine of the one/zero constants, reusing previous location info inappropriately.  Try not caching that and see if it fixes it -- nope.
```
Breakpoint 1, main () at simpleless.silly:4
4       i1 = TRUE;
(gdb) n
5       j1 = FALSE;
(gdb)
6       PRINT i1;
(gdb)
1
7       PRINT j1;
(gdb)
0
9       b = i1 < j1;
(gdb)
10      PRINT b;
(gdb)
9       b = i1 < j1;
(gdb)
10      PRINT b;
(gdb)
0
12      b = j1 < i1;
(gdb)
13      PRINT b;
(gdb)
12      b = j1 < i1;
(gdb)
13      PRINT b;
(gdb)
1
__libc_start_call_main (main=main@entry=0x400470 <main>, argc=argc@entry=1, argv=argv@entry=0x7fffffffdc08) at ../sysdeps/nptl/libc_start_call_main.h:74
74        exit (result);
```

-- see the line numbers jump around.  probably want getLocation(, true) for `ctx->getStop()` in some places (like end of program.)

----------------------------------
