# Changelog V9 (WIP)

## Major Features

### Scope and Variable Lifetime Fix
- **CRITICAL FIX**: Resolved variable scope leak bug where declarations inside `IF`, `ELSE`, `ELIF`, and `FOR` blocks were incorrectly leaking into outer scopes
- Implemented proper scoped variable management with automatic cleanup when exiting control flow blocks
- Variables now correctly shadow outer scope variables and are properly destroyed when leaving their scope

### Symbol Table Removal
- Removed `sym_name` attribute from `silly::DeclareOp`, eliminating symbol table-based variable lookup
- Transitioned to SSA-based variable handling using operation pointer mapping
- Variables now use SSA values directly, improving MLIR compliance and optimization potential
- Removed all of: CallOp, ScopeOp, ReturnOp, YieldOp and their lowering classes and the two pass lowering.

### Parameter Handling Improvements
- **BREAKING CHANGE**: Function parameters now handled as pure SSA values without intermediate storage
- Removed `param_number` attribute from `silly::DeclareOp`
- Parameters accessed directly through `%arg0`, `%arg1`, etc. without requiring `LoadOp`
- Added `silly::DebugNameOp` for parameter debug information attachment

### Debug Information Architecture
- Introduced `silly::DebugNameOp` as unified mechanism for attaching variable names to SSA values
- Supports debug info for:
  - Local variables (via `AllocaOp`)
  - Function parameters (via block arguments)
  - FOR loop induction variables (via SSA values)
- Example output:
  ```mlir
  %0 = "silly.declare"() : () -> !silly.var<i32>
  "silly.debug_name"(%0) <{name = "x"}> : (!silly.var<i32>) -> ()
  "silly.debug_name"(%arg0) <{name = "v"}> : (i32) -> ()
  ```

## MLIR Infrastructure Changes

### Dialect Improvements
- **Removed** `sym_name` attribute from `silly::DeclareOp`
- **Removed** `param_number` attribute from `silly::DeclareOp`
- **Removed** `Symbol` trait from `silly::DeclareOp`
- `silly::DeclareOp` now returns only `!silly.var` type without symbol table integration
- Enhanced `silly::DebugNameOp` to handle multiple SSA value sources

### Parser Architecture Refactoring
- Introduced `PerFunctionState` class to encapsulate per-function parsing state:
  - Induction variable tracking
  - Insertion point stack management
  - Variable declaration maps (by SSA value, not symbol name)
- Added `MlirTypeCache` helper class to organize MLIR type references
- Moved `emitUserError` and `errorCount` to `DriverState` for broader accessibility
- Implemented scope-aware variable management with `enterScope`/`exitScope` callbacks

### Lowering Infrastructure Changes
- Renamed `loweringContext.hpp::PerFunctionState` to `PerFunctionLoweringState` (conflict resolution)
- Replaced symbol-based lookup with operation-pointer-based mapping:
  - **Removed**: `std::unordered_map<std::string, mlir::Operation*> symbolToAlloca`
  - **Added**: `std::unordered_map<mlir::Operation*, mlir::Operation*> declareToAlloca`
- Added helper methods:
  - `LoweringContext::getAlloca(funcName, declareOp)` - retrieve `AllocaOp` for a `DeclareOp`
  - `LoweringContext::setAlloca(funcName, declareOp, allocaOp)` - register mapping
- **Removed** `lookupLocalSymbolReference()` and `createLocalSymbolReference()`

## Grammar & Parser Changes

### Grammar Enhancements
- Added `scopedStatements` rule for control flow blocks
- Scope boundaries now explicitly defined for:
  - `FOR` loops
  - `IF` / `ELIF` / `ELSE` branches
  - Function bodies
- Enables proper variable lifetime management through `exitScope` callbacks

### Parser Improvements
- Organized `parser.cpp` functions by logical grouping for better maintainability
- Constructor for `PerFunctionState` moved to `.cpp` file (workaround for constructor not running when in header)
- Added explicit construction for all `ParseListener` private members
- Implemented `PerFunctionState::isVariableDeclared()` helper
- Removed `biggestTypeOf` as `PerFunctionState` member (scoped to function instead)

## Lowering & Code Generation

### DeclareOp Lowering Changes
- **Removed** `constructVariableDI` call from `DeclareOpLowering` (debug info now via `DebugNameOp`)
- `DeclareOp` lowering now maps operation pointer → `AllocaOp` instead of symbol name → `AllocaOp`
- Type converter integration attempted but not fully functional (TODO for future refactoring)

### DebugNameOp Lowering Implementation
- Properly implemented `DebugNameOpLowering` supporting:
  - Variable declarations (`!silly.var` → `!llvm.ptr`)
  - Function parameters (block arguments)
  - Induction variables (SSA values)
- Generates appropriate LLVM debug intrinsics (`llvm.dbg.declare`, `llvm.dbg.value`)

### Error Handling Improvements
- Added `DriverState::emitInternalError()` - dedicated internal error reporter
- Removed `internal=true/false` boolean parameter from most error message calls
- Cleaner separation between user errors and compiler bugs

## Testing & Quality

### Test Coverage Expansion
- **Added**: `tests/endtoend/failure/error_bad_return_for.silly` - validates return-inside-FOR error detection
- **Added**: `tests/endtoend/failure/error_bad_return_if.silly` - validates return-inside-IF error detection

### Test Infrastructure Updates
- Updated `testit` script:
  - Modified `check_decl_order()` to work without `sym_name` attribute
  - Two-phase approach: collect `silly.declare` SSA values, then map to names via `silly.debug_name`
  - Example:
    ```perl
    # Old: searched for sym_name="varname"
    # New: matches %0 = "silly.declare" then maps %0 to name via debug_name
    ```
- Removed `sym_name` references from all `tests/dialect/*` test cases
- Enhanced test cleanup: deletes all output artifacts per test stem

### Bug Fixes
- **CRITICAL**: Fixed variable scope leak allowing control flow block variables to persist beyond their intended lifetime
- Fixed constructor initialization issues with `PerFunctionState` (moved from header to `.cpp`)
- Fixed name collision between parser and lowering `PerFunctionState` classes

## Documentation Updates

### Code Documentation
- Multiple doxygen comment fixes across codebase
- Improved inline comments explaining SSA-based variable handling

## Known Issues & Limitations

### Type Converter Integration
- Type converter added to `LoweringContext` but not fully functional for `replaceOp()` in `DeclareOp` lowering
- Currently using operation-pointer mapping as workaround
- **TODO**: Properly implement type conversion for `!silly.var<T>` → `!llvm.ptr` to enable automatic operand updates

### ScopeOp Redundancy
- `silly::ScopeOp` still present in dialect but now redundant given scope-aware parser
- **TODO**: Consider removing `ScopeOp` entirely in future version

## Migration Notes

### For Compiler Developers
- **Symbol table eliminated**: If you reference `declareOp.getName()`, migrate to operation pointer-based lookup
- **Parameter handling changed**: Parameters no longer create `DeclareOp` with `param_number`; use direct SSA references
- **Debug info**: Use `silly::DebugNameOp` instead of `sym_name` attribute on `DeclareOp`

### MLIR Output Changes
Before V9:
```mlir
%0 = "silly.declare"() <{sym_name = "x"}> : () -> !silly.var<i32>
%1 = "silly.declare"() <{param_number = 0 : i64, sym_name = "v"}> : () -> !silly.var<i32>
```

After V9:
```mlir
%0 = "silly.declare"() : () -> !silly.var<i32>
"silly.debug_name"(%0) <{name = "x"}> : (!silly.var<i32>) -> ()
"silly.debug_name"(%arg0) <{name = "v"}> : (i32) -> ()
```

### Test Suite Changes
- Tests checking `sym_name` attribute will fail
- Update declaration order tests to use new two-phase SSA+`debug_name` pattern
- Scope leak tests now properly validate variable lifetime

---
