## TODO

### running list of other issues and ideas, semi-randomly ordered
----------------------------------

#### misc
* tests/endtoend/expressions/modfloat.silly broken with mix of float32/float64's
* Run include-what-you-use on lowering.cpp (now that LoweringContext.cpp has been split out.)  Will probably have to build it.
* Also using old ctest/testit:

    - dwarfdump
    - get
    - mlirparsetest

#### Driver

* don't think that driver is removing outputs before trying to recreate, so if there is an error after success, it is not visible.
* any driver error should delete any files opened (.o, .s, .ll, .mlir, ...) -- do I need callbacks, or is LLVM taking care of that?
* implement -o option for the exe name.
* If user doesn't specify -c, then the created .o file should go to a /tmp/ or TMPDIR path allocated using mkstemps, and removed post link.

#### Grammar

* Add MODULE, MAIN, INTERFACE statements.  MODULE .silly's should have only FUNCTION.  INTERFACE to have only prototypes.
* Add FUNCTION declaration syntax (for use in external MODULE objects.)

#### Lowering

* Attempted introducing a type convertor to DeclareOp and DebugNameOp lowering, but couldn't get it to work (i.e.
  all of DebugNameOp, LoadOp, AssignOp reference the DeclareOp, but once lowered, want the DeclareOp SSA value to be
  swapped out for the alloca SSA value).  Couldn't get that to work, and have reverted to an unordered_map (but now
  keyed on the DeclareOp's Operation*, not a function-name::variable-name pair, as was the case for the symbol based
  implementation.

  Followup on that idea -- it makes more sense to use existing infrastructure.

#### diagnostics

* implement error numbers/classes.  give the option of also showing suppressed internal errors.
* implement gcc like limit on the number of errors.

#### Driver

* add testing for: .o inputs to the driver; silly -c empty.silly ; silly empty.o
* consider OO structure for driver.cpp
* tried to use mlir::OwningOpRef<mlir::ModuleOp> for ParseListener::getModule (and the mod op it contains), but
  couldn't get it to work and reverted all such experimentation.  This means that I leak the mlir::ModuleOp
  and it only gets freed implicitly on return from main.  That's clumsy, and would probably show up as
  a valgrind leak.  Revisit this separate from trying to add the .mlir read+parse.

#### misc
* Have an effective lexical scope for loop variables, but am emitting DI for them at a function scope.  This will probably do something weird if a loop variable is used in multiple loops.
* Grammar allows for function declared in a function, now prohibited (`error_nested.silly`).  This wouldn't be too hard to fix, see notes in parser.cpp (enterFunction), however, scoping rules for function lookup would have to be decided.
* Forward declarations for functions?
* Allow: `INT64 a = 1, b = 2, c = 3;` (`chained_comparison_parens.silly`)?, or `INT64 a{1}, b{2}, c{3};`
* [make] /build/ is hardcoded in these places:

```
bin/build
tests/dialect/lit.cfg.py:11:config.test_exec_root = os.path.join(config.test_source_root, "..", "..", "build", "tests", "dialect")
tests/dialect/lit.cfg.py:15:    os.path.join(config.test_source_root, "..", "..", "build", "lit.site.cfg.py")
```

(should be able to configure and build from anywhere -- even out of tree.)

* Would be good to add a CALL verify that checks if the function has a return, to make sure it is not used as a standalone statement without assignment.
* lowering error handling: Review all the notifys -- emitError may be more appropriate in some places.
* `div_zero_int` -- different results on intel vs. arm.
* `negative_step_for.silly` -- would be better to put in a (perhaps optional) runtime check for negative or zero step sizes in FOR statements.  test case for the zero step condition: `zero_step_for.silly` -- not included in automation, as it infinite loops (would be better if it did not.)
* `error_invalid_unary` -- regression by tweaking the test. was triggering on y undeclared, not on the parse error -- which doesn't actually drive a compile error!
* Forgetting RETURN in `array_elem_as_arg.silly` has a very confusing error.
* Need a sema pass: For example, initializer-list shouldn't reference variables, only constant-expressions, or expressions with parameters.  t/c for this: `error_nonconst_init.silly`
* Need tests for debug capability.  For example, new induction variable support (t/c: `for_simplest.silly` -- only tested manually.)  Start with at least generalizing the test hacks in testit.  Example addition, also check for:

```
  DW_AT_type                  <0x0000006e> Refers to: int64_t
```

  in the `for_simplest` t/c.
* SSA form for loop variable access didn't fix the gdb line number ping pong in loop body line stepping.  Simpler t/c: printdi.silly -- Example:

```
  (gdb) b 6
  Breakpoint 2 at 0x400798: file printdi.silly, line 6.
  (gdb) c
  The program is not being run.
  (gdb) run
  Starting program: /home/peeter/toycalculator/tests/endtoend/out/printdi
  [Thread debugging using libthread_db enabled]
  Using host libthread_db library "/lib64/libthread_db.so.1".

  Breakpoint 2, main () at printdi.silly:6
  6           t = c[i];
  (gdb) n
  8           PRINT "c[", i, "] = ", t;
  (gdb) n
  6           t = c[i];
  (gdb) p i
  $1 = 0
  (gdb) n
  8           PRINT "c[", i, "] = ", t;
  (gdb) n
  c[0] = 1
  4       FOR (INT32 i: (0, 5))
  (gdb) p i
  $2 = 0
  (gdb) n

  Breakpoint 2, main () at printdi.silly:6
  6           t = c[i];
```

  (see that we have two hits to PRINT and the assignment before the breakpoint hits again.)


----------------------------------
### Bugs
* `error_intarray_bad_constaccess.silly` should fail but doesn't.
* Have ABORT builtin now to implement failure codepath for runtime array range checking.  Could also use this for assignment to bool from source value > 1.
* line stepping over a multi-argument print statement is getting a new line entry for each individual runtime call:

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
  but still have this issue in arrayprod.  Using unknownLoc for the print args doesn't help -- tried as an experiment.  What's probably required is a fusedLoc for all elements that contribute to a multiple instruction sequence that's tied to a single line (always do that, or perhaps just for PRINT?)

```
  %8 = bitcast ptr %1 to ptr, !dbg !24
  store { i32, i32, i64, ptr } { i32 1, i32 0, i64 42, ptr null }, ptr %8, align 8, !dbg !24
  call void @__silly_print(i32 1, ptr %1), !dbg !24
```

* In initFill: `loc = rewriter.getUnknownLoc();` This was a HACK to suppress location info for these implicit memset's, so that the line numbers in gdb don't bounce around.  I think that the re-ordering that I now do in the DeclareOp builder is messing things up, and that is the root cause of the trouble (see that in arrayprod.silly trying to step over the dcls if this hack is removed.)

----------------------------------
### Test:

* More testing for FOR including (dcl, assignment, call, IF/ELIF/ELSE, ...).
* Tests for all the type conversions (i.e.: binary and unary arith operators)
* More tests for array access and assignment.  Example, some vector expressions (add, convolution, ...).
* Tests for call in binary and unary expressions.

----------------------------------
### Expressions:
* Assignment like initialization syntax is not available for array variables.  That should trigger an error.
* Maybe: Get rid of CALL and support calls in binary expressions: r = v * foo().
* Expressions that aren't parsed properly (like `CALL factorial(v - 1)` used to) lead to mysterious seeming parse error: Should do better.
* NOT should be allowed for boolean expressions in IF/ELIF predicates.
* no test case for CALL in booleanExpression
* grammar allows for CALL to have CALL in the parameter list.  Not tested.
* PRINT now supports rvalue-expressions.  I had to hack it to interpret INT8 [] arrays as strings in parseRvalue (for PRINT/ERROR handling cases only), as strings aren't a first class entity -- fix that.

----------------------------------
### Maintainance:

* All the runtime functions should take location context to show where in the users code the error was, if one happens (i.e.: GET functions)
* Move scf lowering into 1st pass?  Attempt at this in xpgn:streamline-passes-attempt branch (not pushed.)
* Lots of cut and paste duplication for type conversion in lowering.cpp -- split out into helper functions.
* Final version of ELIF support meant that I didn't need getStopLocation() anymore (was using it to generate scf.yield with the end of the block location).
* merge these?

----------------------------------

### Exit/return
* Maybe: enforce i8 Exit type in the grammar (i.e.: actual UNIX shell semantics.)  Don't have an ExitOp anymore, just ReturnOp, but could switch from the hardcoded i32 return type in the builder.
* Allow EXIT at more than the end of program (that restriction is currently enforced in the grammar.)
* Allow return/exit at other places in a function/program so get out prematurely.
* Remove the mandatory RETURN in the FUNCTION grammar, and mirror the exitHandled

----------------------------------

### String tests and behaviour:
* longstring.silly: compiler truncates an assignment from a too long string literal.  Could implement safe vs. checked behaviour so that such an assignment would trigger an error instead.

----------------------------------
### Control flow:
* Should implement a break-like keyword for the FOR loop.  That would allow for a "poor man's while", with an effectively infinite very large loop bound.
* AND/OR: Implement WHILE/DO/BREAK/CONTINUE statements.

----------------------------------
### Tablegen/MLIR:
* Don't have any traits defined for my MLIR operations (initially caused compile errors, and I just commented-out or omitted after that.)
* Validators, and better custom assembly printers.
* MLIR has considerable capability for semantic checking, but I'm not exploiting that here, and have very little in custom verifiers.
* If verify fails, I shouldn't proceed to lowering.  Do I?

----------------------------------
### arrays:
* implement runtime bounds checking (make it a compiler option?)
* attempt to put in constant array access range checking did not work (for AssignOp lowering, probably also for LoadOp lowering -- untested).

----------------------------------
### Misc:
* GET into a BOOL should logically support TRUE/FALSE values, and not just 0/1.  PRINT of a BOOL should display TRUE/FALSE, not just 1/0.
* Write a MLIR walker (and/or opt-silly front end) to see how to answer code questions about a given program.
  Perhaps do it as an mlir pass, now that I have a libSillyDialect.so available for use with mlir-opt.
* CAST operators.  Could also implement that with a "standard library", or on demand.
* Just for fun: Implement a JIT so that the "language" has an interpretor mode, as well as static compilation.

----------------------------------
## Debugging:

* Expand Dwarf DI test cases (have just one in `bin/testit` for `tests/endtoend/f.silly`, and it is completely manual, and only checks bar0 line number.)

* t/c: function.silly: see if this is still and issue, and debug it, if it is.

  Line stepping behaves wrong after CALL too, showing the wrong line after some calls:

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
* gdb session for simpleless.silly is not behaving right with respect to 'next'.  Suspect that this is due to my cachine of the one/zero constants, reusing previous location info inappropriately.  Try not caching that and see if it fixes it -- nope.

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

  See the line numbers jump around.  probably want getLocation(, true) for `ctx->getStop()` in some places (like end of program.)

----------------------------------
