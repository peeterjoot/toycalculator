# Changelog: V8 Release (WIP)

(These notes are up to and including fd383d8e1475a0bd40bc5977339404db51d24839)

## Major Features

### Expression System Overhaul

**Generalized Expression Support**
- Implemented full operator precedence and associativity
- Support for parentheses and expression nesting
- Chained unary operators (e.g., `- - x`, `NOT NOT condition`)
- NOT is now restricted to integer types
- Complex expressions: `PRINT (1 + 2 * 3) * 4 - 1`
- Expressions now allowed in:
  - FOR loop range values: `FOR (INT32 i : (+a, -b, c + z))`
  - Function call parameters: `CALL factorial(v - 1)`
  - Array indices: `arr[i + 1]`, `arr[CALL someFunc()]`
  - Assignment right-hand sides
  - Return and exit statements: `EXIT 39 + 3`
  - Initializer lists (parameters and constants)

**Operator Grammar Refactoring**
- Introduced dedicated operator rules to provide clean vectors in visitor pattern
- Example: `multiplicativeOperator : TIMES_TOKEN | DIV_TOKEN`
- Fixes ambiguous arrays of terminal nodes
- Enabled previously failing test: `expression5.silly`

**Type Handling Improvements**
- Fixed type promotion semantics: computations now occur with proper type inference
- Eliminated premature type conversions in expression evaluation
- Constants created with correct type from context (no more i64 truncation everywhere)
- Floating-point to integer conversions use floor operation (not rounding)

### Variable Declaration & Initialization

**Declaration with Initialization**
- New syntax: `INT32 x = 42;` and `INT32 x{42};`
- Support for all scalar types (INT8/16/32/64, FLOAT32/64, BOOL)
- Initialization expressions can reference:
  - Constants and literals
  - Parameters (in function scope)
  - Previously declared variables
  - Function calls: `INT32 z[1]{CALL answer()}`
  - Complex expressions: `INT32 neg[1]{-(10 + 5)}`

**Array Initializer Lists**
- C++-like syntax: `INT32 arr[3]{1, 2, 3}`
- Shorter lists are zero-padded
- Excess initializers trigger compile error
- Type conversions applied: `INT32 a[2]{3.9, 4.2}` → `{3, 4}`
- String initialization: `STRING msg[6] = "Hello"`, `STRING msg[6]{"Hello"}`
- Test cases: `initlist`, `initarray`, `error_init_list1/2`

**Auto-Initialization Option**
- New `--init-fill nnn` command-line option
- Default: zero-initialize all variables (unless initializer-list initialized, which always uses zero padding.)
- Custom fill values (0-255) for debugging
- Test case: `arrayprod.silly` with fill=255

### Control Flow Enhancements

**FOR Loop Improvements**
- SSA form for loop bodies (removed unnecessary assign operations)
- Induction variables now proper loop arguments
- Type-safe induction variables: `FOR (INT32 i : (1, 10))`
- Expression support for range values
- Debug instrumentation for induction variables
- Test cases: `for_simplest`, `nested_for`, `negative_step_for`

**IF/ELIF/ELSE Fixes**
- Fixed insertion point logic for nested if/elif/else
- Corrected MLIR generation for ELIF chains
- Bug fix: `minimal_eliftest.silly` duplicate output issue
- Test coverage: `nested_if_deep`, `nested_if_elif_else`

### I/O and Error Handling

**Enhanced PRINT/ERROR**
- Multi-argument support: `PRINT "x = ", x, ", y = ", y`
- Expression support: `PRINT 40 + 2, -x, f[0], CALL foo()`
- New `CONTINUE` modifier to suppress newline
- Unified runtime implementation (single function call with struct array)
- ERROR statement: prints to stderr instead of stdout

**New ABORT Statement**
- Prints `<file>:<line>:ERROR: aborting` message to stderr
- Terminates program immediately
- Example: `ERROR "Fatal error"; ABORT;`
- Lowering generates call to `__silly_abort(filename, line)`

### Function Improvements

**Return Statement Enhancements**
- Return expressions: `RETURN a + b`
- Fixed BOOL-returning functions with bare `RETURN`
- Eliminated dummy return rewrite code remnants
- Better error messages for type mismatches

**Call Expression Integration**
- CALL can appear in any expression context
- Unary expressions: `x = - CALL foo()`
- Binary expressions: `x = 1 + v * CALL foo()`
- As function parameters: nested calls supported
- In predicates and boolean expressions

## MLIR Infrastructure Changes

### Dialect Improvements

**SSA Form Refactoring**
- New `Silly_VarType` for abstract variable locations
- DeclareOp now returns `!silly.var<type [shape]>` handle
- AssignOp and LoadOp reference SSA values instead of symbols
- Example: `%0 = silly.declare {sym_name = "x"} : <i64>`
- Example: `silly.assign %0 : <i64> = %c42_i64 : i64`
- Custom type printer: `<i64>` instead of `<i64 []>` for scalars

**Debug Information**
- New `silly.debug_name` operation for induction variables
- Granular location tracking for expression elements
- DWARF instrumentation for loop variables
- Example location info for `PRINT "c[", i, "] = ", t` shows separate locations for each element

**Dialect as Shared Library**
- Packaged as `libSillyDialect.so` (no longer statically linked)
- Can be loaded by `mlir-opt` as a plugin
- Removed custom DeclareOp asm printer (default is better)
- Better integration with MLIR ecosystem

### Verification & Testing

**LIT Test Infrastructure**
- Added lit-based testing for dialect operations
- FileCheck integration for MLIR validation
- Test directory: `tests/dialect/`
- Coverage for verify() functions:
  - DeclareOp: parameter validation, type checking
  - ReturnOp: type compatibility, scope validation
  - ScopeOp: must be in function
- Example tests: `bad_return_*.mlir`, `bad_declare_*.mlir`

**CTest Integration**
- Migrated from manual `testit` script to CTest
- End-to-end tests organized hierarchically:
  - `tests/endtoend/array/`
  - `tests/endtoend/bool/`
  - `tests/endtoend/operators/`
  - etc.
- Dialect verification tests: `tests/dialect/`
- Test properties: labels, timeouts, expected output patterns

## Build System & Project Structure

### Directory Reorganization

**New Source Layout**
```
src/
  ├── runtime/        # libsilly_runtime.so
  ├── dialect/        # libSillyDialect.so
  ├── driver/         # silly compiler (and parser and lowering)
  ├── grammar/        # ANTLR4 grammar
  └── include/        # shared headers
```

**Build Artifacts**
```
build/
  ├── bin/silly          # compiler driver
  ├── lib/
  │   ├── libsilly_runtime.so
  │   └── libSillyDialect.so
  └── [subdirectories matching src/]
```

**Per-Component CMakeLists**
- Each subdirectory has its own CMakeLists.txt
- Top-level CMakeLists.txt is minimal
- Clear separation of concerns
- Symlinks for convenience: `build/bin/silly` → `build/src/driver/silly`

### Build Improvements

**ANTLR4 Integration**
- Grammar moved to `src/grammar/Silly.g4`
- Generated files in `build/src/grammar/SillyParser/`
- Custom target: `GenerateANTLR`
- Proper dependency tracking

**Linker Configuration**
- Runtime library path (relative to bin/silly): `-Wl,-rpath,../lib`

**Command-Line Options**
- `--verbose-link`: Show linker command (implicit on failure)
- Existing options maintained and documented

### Install configuration
- Added cmake `SILLY_ENABLE_INSTALL` rules (bin/build configures the install path to
  /opt/silly/)

## Grammar & Parser Changes

### Grammar Cleanup

**Consistent Naming Convention**
- Statement suffixes added for clarity:
  - `abort` → `abortStatement`
  - `ifelifelse` → `ifElifElseStatement`
  - `for` → `forStatement`
  - `print` → `printStatement`
  - `assignment` → `assignmentStatement`
  - `declare` → `declareStatement`
  - etc.

**Removed Ambiguities**
- Fixed `(PLUSCHAR_TOKEN | MINUS_TOKEN)?` in literal patterns
- Moved sign handling to `unaryExpression` rule
- Prohibited chained comparisons: `1 < 2 < 3` is error
- Prohibited chained equality: `1 EQ 1 NE 1` is error

**Type Rules**
- Introduced `intType` rule for reuse
- Used in `intDeclareStatement` and `forStatement`
- Better grammar organization and maintainability

### Parser Improvements

**Code Organization**
- Renamed `MLIRListener` → `ParseListener` (more accurate)
- Helper functions for constant creation:
  - `parseBoolean()`
  - `parseInteger()`
  - `parseFloat()`
- Consistent parameter ordering: `loc` first in most functions
- Doxygen comments for private functions
- Prohibit nested functions.  t/c: `error_nested.silly`.
- Error handling and error message printing completely overhauled.  There's no
  more use of exception handling classes.

**State Management**
- Removed: `currentAssignLoc`, `callIsHandled`, `assignmentTargetValid`
- Removed: `currentVarName`, `currentIndexExpr`, `varStates`
- Simplified control flow with fewer state variables
- Induction variable stack: `vector<pair<string, Value>>`
- Generate gcc/clang style error output with context.  Example:
```
> silly error_nested.silly
error_nested.silly: In function ‘foo’:
error_nested.silly:5:5: error: Nested functions are not currently supported.
    5 |     FUNCTION bar( INT32 v )
      |     ^
error_nested.silly:14:12: error: Undeclared variable r (symbol lookup failed.)
   14 |     RETURN r;
      |            ^
> echo $?
3
```

**Insertion Point Handling**
- Fixed insertion point stack management
- Proper handling for nested FOR loops
- Fix for nested IF/ELIF/ELSE blocks
- Declaration hoisting to scope beginning

## Lowering & Code Generation

### Error Handling Improvements

**MLIR-Idiomatic Error Reporting**
- Replaced `assert()` with `rewriter.notifyMatchFailure()`
- Replaced `throw` with `mlir::LogicalResult`
- Helper functions return `mlir::failure()` on error
- Example pattern:
  ```cpp
  if (mlir::failed(lState.createGetCall(...))) {
      return mlir::failure();
  }
  ```

### Runtime Simplification

**PRINT Runtime Consolidation**
- Previous: separate call per argument
- New: single `__silly_print()` call with struct array
- Struct layout: `{kind, flags, data, ptr}`
- Array size: maximum print arguments in function
- Reduced LLVM-IR verbosity significantly (but with tradeoffs, as there's now
  stack gorp to be filled in for each `PRINT` call)

**Type Handling**
- Standardized helper function signatures: `(loc, rewriter, ...)`
- Split out common functionality:
  - `infoForVariableDI()` for debug info construction
  - `constructInductionVariableDI()` for loop variables
  - `generateAssignment()` for use in multiple contexts

## Language Feature Removals

**Deprecated DECLARE Statement**
- `DCL`/`DECLARE` removed from grammar and parser (see Migration Notes)
- Replaced with explicit type declarations
- All tests updated

## Documentation Updates

### README Improvements

**Language Overview**
- Comprehensive examples for each feature
- Clear syntax documentation
- Operator precedence table
- Expression grammar specification

**Operations Reference**
- Detailed description of all operations
- Example code for each statement type
- Notes on limitations and quirks

**Building & Testing**
- Updated build instructions
- CTest usage examples
- Tool documentation (`silly-opt`)

### Changelog Organization

**New Structure**
- Split into `Changelog.md` and `Changelog.old.md`
- Hierarchical organization by feature area
- Cross-references to test cases
- MLIR examples for major changes

## Testing & Quality

### Test Coverage Expansion

**New Test Categories**
- Expression tests: complex precedence, parentheses, type mixing
- Initialization tests: various expression types in initializers
- Control flow: nested loops, nested if/elif/else
- Error cases: comprehensive negative testing
- Debug info: DWARF validation tests

**Test Organization**
```
tests/
  ├── endtoend/
  │   ├── array/
  │   ├── bool/
  │   ├── operators/
  │   ├── expressions/
  │   └── ...
  └── dialect/
      └── *.mlir
```

**Test Case Examples**
- `factorial.silly`: recursion test
- `arrayprod.silly`: array operations with explicit `--init-fill 255`
- `minmax.silly`: helper functions (found parser bug)
- `for_simplest.silly`: minimal FOR with debug info
- `nested_for.silly`: loop nesting
- `error_*.silly`: comprehensive error coverage

### Bug Fixes

**Critical Fixes**
- `minimal_eliftest.silly`: duplicate output in ELIF chains
- `nested_for.silly`: return statement in wrong scope
- `minmax.silly`: predicate generation bug (x,x instead of x,y)
- Array index type casting from any integer width
- Insertion point management for nested constructs

**Type System Fixes**
- Type promotion in assignments corrected
- Constant type inference from context
- No more spurious i64 truncations in MLIR

## Tools & Utilities

### New Tools

**silly-opt**
- Wrapper for `mlir-opt` with silly dialect loaded
- Simplified MLIR testing and debugging
- Pretty-printing support
- Example: `silly-opt --pretty --source out/loadstore.mlir`

**Updated testit**
- `--clean` option for output cleanup
- `--driverpath` for flexible cmake build directory configuration
- Better integration with CTest
- DWARF validation with `dwarfdump`
- Remove all the test lists (success and failure cases) -- moved to ctest.

### Build Scripts

**Improvements**
- Build directory no longer hardcoded (mostly)
- `bin/silly-opt` and `bin/testit` now portable
- Symlink creation for convenience (exposing install like hierarchy in cmake build dir.)

## Known Issues & Limitations

**Documented Quirks**
- Variables declared in FOR/IF persist beyond block scope
- FOR step must be positive (no runtime check)
- RETURN statement must be last in function
- RETURN statement is mandatory
- EXIT must be at end of program
- BOOL storage: one byte per element (arrays not bit-packed)
- GET to BOOL only accepts 0 or 1 (no TRUE/FALSE)
- FOR loops with negative ranges have undefined behaviour

## Migration Notes

Probably pointless to state, since I'm surely the only user, but, ...

**V7 to V8**

1. **Replace DECLARE:**
   ```bash
   perl -p -i -e 's/\bDCL\b/FLOAT64/g' *.silly
   perl -p -i -e 's/\bDECLARE\b/FLOAT64/g' *.silly
   ```

2. **FOR loops now require variable declaration:**

   ```
   // Now:
   FOR (INT32 i : (1, 4, 1))
   // instead of:
   INT32 i;
   FOR (i : (1, 4, 1))
   ```

2. **Update build:**
   - Clear old build directory
   - New directory structure: check artifact locations
   - Symlinks created automatically in `build/bin/` and `build/lib/`

3. **Test execution:**
   - Use `ctest` instead of manual `testit` runs
   - Test categories available: `ctest -L array`, etc.
   - testit requires libdwarf-tools now for dwarfdump

---

<!-- vim: set tw=80 ts=2 sw=2 et: -->
