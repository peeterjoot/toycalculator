# Changelog V0.10.0 (WIP)

## LLVM version

* Updated to llvm-project 22.1.0, with corresponding changes to the builder.create<> syntax and a few other updates.

## Major Features

### MODULE import support.

Have a primative, but working module import system:

* must name that module with --imports.

Processing algorithm:

- The --import module is compiled as far as MLIR.  This module could
  also be a silly dialect mlir text file, or a mlirbc binary (both of
  those untested.)  It can't be .ll, nor .o
- Then the rest of the silly sources are compiled (all the way to object
  code.)  If there are any IMPORT statements in those sources, they are
  processed by creating FuncOp declarations from all the FuncOp's
  implemented in the --import named file.
- Then we go back to the the --import named file, and finish lowering it
  down to LLVM-IR and object code.
- Link everything.

Demo:
```
fedoravm:/home/peeter/toycalculator/tests/endtoend/driver> cat callmod.silly
IMPORT callee; // foo, bar

PRINT "In main";
CALL bar( 3 );
PRINT "Back in main";

INT32 rc = CALL foo();

PRINT "Back in main again, rc = ", rc;
fedoravm:/home/peeter/toycalculator/tests/endtoend/driver> #ls -ltr
fedoravm:/home/peeter/toycalculator/tests/endtoend/driver> cat callee.silly
// callee.silly
MODULE;

FUNCTION bar ( INT16 w )
{
    PRINT w;
    RETURN;
};

FUNCTION foo ( ) : INT32
{
    CALL bar( 3 );

    RETURN 42;
};
fedoravm:/home/peeter/toycalculator/tests/endtoend/driver> rm -f callmod ; silly --imports callee.silly callmod.silly
fedoravm:/home/peeter/toycalculator/tests/endtoend/driver> ./callmod
In main
3
Back in main
3
Back in main again, rc = 42
```

* Allow IMPORT within MODULE.

### Debug Information Architecture

* Switched to DWARF5.  Now using fused locations for FuncOp's, and have infrastructure that allows for using them in other places (that is disabled, pending a for-sure use-case, since that results in columnar location info loss.)
* LexicalBlockAttrDI implemented for IF, ELSE, ELIF (but not yet for FOR induction variables.)

## General Changes
* Switched from std::format to llvm::formatv for most calls to llvm::errs(), dbgs(), outs().

## MLIR Infrastructure Changes

### Dialect Improvements

### Parser Architecture Refactoring

### Lowering Infrastructure Changes

## Grammar & Parser Changes

* Remove the old DCL_TOKEN from the antlr4 grammar

### Experimental BISON/FLEX front end and grammar.

* Very incomplete.

### Grammar Enhancements

### Parser/builder Improvements
* I didn't have a good reason to set the IP to beginning of function for declarations anymore, now that the
  symbol dependencies (old silly::ScopeOp) is gone.  Do them in place instead.

## Lowering & Code Generation

### Error Handling Improvements

### Code structure

## Driver

* Implemented --no-verbose-parse-error.
* Implemented --emit-llvmbc, and round trip support for both .ll and .bc
* Implemented support for -o w/ --emit-llvm, --emit-mlir, ... (also fixing path construction when fully qualified paths used.)
* Fix -o exename w/ --output-directory (adds a test for that, also testing --keep-temp)

## Testing & Quality

### Test Coverage Expansion

* Moved tests/dialect to tests/lit/dialect
* Added tests/lit/driver, with round trip tests for -c (--emit-llvm, --emit-mlir, ...)
* Merged tests/endtoend/mlirparsetest and tests/endtoend/debug into tests/lit/debug (all now using more sensible single source lit/FileCheck orchestration)
* test coverage for various driver error codepaths.
* Moved tests/endtoend/driver/* to tests/lit/driver/ (renaming each for clarity with lit conversion)
* Moved tests/endtoend/failure/* to tests/lit/syntax/
* Moved tests/endtoend/exit to tests/lit/exit
* Moved tests/endtoend/get to tests/lit/get
* Moved tests/endtoend/fatal to tests/lit/fatal (remove RunFailureTest.cmake)
* Moved tests/lit/* to tests/, moved tests/endtoend/* to Samples/, generate lit wrappers for all the Samples/*/*silly
* Did a systematic rename of all the tests/, using names that match the testing better, and removing redundancy.
* Range check tests for too big int/float literal: `toolong-int-literal.silly`, `toolong-float-literal.silly` (also catching std::exception)

### Test Infrastructure Updates

### Bug Fixes

## Documentation Updates

### Code Documentation

## Known Issues & Limitations

### Type Converter Integration

## Migration Notes

### For Compiler Developers

### MLIR Output Changes

### Test Suite Changes

---
