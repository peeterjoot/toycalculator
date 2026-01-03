## TODO

----------------------------------
* Test:

  - a whole range of statements in IF-then and if-ELSE blocks, and FOR (dcl, assignment, call, ...)
  - tests for all the type conversions (i.e.: binary and unary arith operators)

----------------------------------
* Maintainance:
  - lowering error handling is pschizophrenic: mix of: assert(), throw, llvm::unreachable, rewriter.notifyMatchFailure, emitError, ...
  - Move scf lowering into 1st pass?  Attempt at this in xpgn:streamline-passes-attempt branch (not pushed.)
  - Lots of cut and paste duplication for type conversion in lowering.cpp -- split out into helper functions.

----------------------------------
* PRINT.  Allow a list of values, instead of just one (print all to one line.)
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
  - AND/OR: Implement ELIF/WHILE/DO/BREAK/CONTINUE statements.

----------------------------------
* Tablegen/MLIR:
  - Don't have any traits defined for my MLIR operations (initially caused compile errors, and I just commented-out or omitted after that.)
  - Validators, and better custom assembly printers.
  - MLIR has considerable capability for semantic checking, but I'm not exploiting that here, and have very little in custom verifiers.
  - If verify fails, I shouldn't proceed to lowering.  Do I?

----------------------------------
* arrays:
  - implement runtime bounds checking (make it a compiler option?)
  - attempt to put in constant array access range checking did not work (for AssignOp lowering, probably also for LoadOp lowering -- untested).  Also want:

    // allow: Now t[i+1] or t[someFunc()], ..., to parse correctly:
    indexExpression
      : ARRAY_START_TOKEN assignmentExpression ARRAY_END_TOKEN
      ;

    but currently have much more limited index expressions:

      : ARRAY_START_TOKEN (IDENTIFIER | INTEGER_PATTERN) ARRAY_END_TOKEN


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
