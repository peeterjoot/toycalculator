## TODO

----------------------------------
* Next TODO:
  - Array initializer syntax: grammar done.  next is parser.  Also write some tests that use it to see if the parser chokes.

----------------------------------
* Bugs:
  - `error_intarray_bad_constaccess.silly` should fail but doesn't.
  - Have ABORT builtin now to implement failure codepath for runtime array range checking.  Could also use this for assignment to bool from source value > 1.

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
  - Get rid of CALL and support calls in binary expressions: r = v * foo()?
  - Implement more complex expressions (chains of operators...)
  - Expressions that aren't parsed properly (like `CALL factorial(v - 1)` used to) lead to mysterious seeming parse error: Should do better.
  - Review the parser... any other places where buildUnary is called that ought to be parseRvalue?
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
