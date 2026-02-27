# Changelog V10 (WIP)

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

### Debug Information Architecture

## MLIR Infrastructure Changes

### Dialect Improvements

### Parser Architecture Refactoring

### Lowering Infrastructure Changes

## Grammar & Parser Changes

### Grammar Enhancements

### Parser Improvements

## Lowering & Code Generation

### Error Handling Improvements

### Code structure

## Driver

* Implemented --no-verbose-parse-error.
* Implemented --emit-llvmbc, and round trip support for both .ll and .bc

## Testing & Quality

### Test Coverage Expansion

### Test Infrastructure Updates

### Bug Fixes

## Documentation Updates

### Code Documentation

## Known Issues & Limitations

### Type Converter Integration

### ScopeOp Redundancy

## Migration Notes

### For Compiler Developers

### MLIR Output Changes

### Test Suite Changes

---
