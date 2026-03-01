## tag: V5 (Dec 22, 2025)

The language now supports functions, calls, parameters, returns, and basic conditional blocks.

### 1. Build / LLVM Version Updates

* Updated build scripts (`bin/build`, `bin/env`) for LLVM path handling.
* Switched project to LLVM/MLIR 21.x (`llvmorg-21.1.0-rc3` and later). Dropped support for ≤ 20.1.8.
* Added a Flang-related patch (`llvm-patches/llvm21.flang.patch`).

### 2. Return / Location Fixes

* Improved location info for implicit/dummy returns (now uses last statement location instead of file start).
* Introduced per-function parser state to track last location.

### 3. Conditional Statements – IF / ELSE

* Preliminary implementation of IF and ELSE grammar and lowering to `scf.if` / `scf.else`.
* Split the old combined `ifelifelse` rule into separate `if`, `elif`, `else` rules (`ELIF` not yet implemented).
* Added `CallOpLowering` (moved calls out of `ScopeOpLowering` because calls like implicit `PRINT` can now appear inside `if/else` blocks).
* Integrated SCF lowering into the second lowering pass.
* Generalized location helpers and predicate parsing in preparation for full conditional support.
* Updated README to note that IF/ELSE is now supported (but needs much more testing, especially nested IFs).
* Reduced `samples/if.silly` to only the currently implemented subset; moved unimplemented parts to `if2.silly`.

### 4. Array element access and assignment (rvalues and lvalues)

* Added support for array element assignment: `t[i] = expr;`
  * Generalized `silly.assign` to take an optional `index` operand (`Optional<Index>`).
  * Updated grammar to allow `scalarOrArrayElement` (variable optionally indexed) on the LHS of assignments.
  * Implemented lowering of indexed `silly.assign` using `llvm.getelementptr` + `store`, with static out-of-bounds checking for constant indices.
  * Added custom assembly format: `silly.assign @t[%index] = %value : type`.

* Added support for loading array elements (rvalues): `x = t[i];`
  * Generalized `silly.load` to take an optional `index` operand.
  * Implemented lowering using `llvm.getelementptr` + `load`, with static bounds checking.
  * Added custom assembly format: `silly.load @t[%index] : element_type` (scalar case prints without brackets).

* Parser and frontend changes:
  * Introduced `scalarOrArrayElement` and `indexExpression` grammar rules.
  * Grammar: Extended unary expression handling and many statement types (PRINT, RETURN, EXIT, call arguments, etc.) to accept array element references.
  * PRINT/RETURN/EXIT of array elements is supported and tested, but lots more testing should be done in other contexts (call arguments, ...).  It is possible that additional builder/lowering work will show up from such testing.
  * Added `indexTypeCast` helper for converting parsed index values to `index` type.
  * Updated `intarray.silly` sample to demonstrate full round-trip (declare → assign element → load element → print).  Also added rudimentary test for unary and binary expressions with array elements, and exit.
  * Updated `function_intret_void.silly` with return of array element.
  * New test `exitarrayelement.silly` to test exit with non-zero array element (zero tested in `intarray.silly`)

* Lowering improvements:
  * Factored out `castToElemType` helper for consistent type conversion during stores/loads.
  * Fixed several bugs during iterative development (GEP indexing, type handling, optional operand creation).

* README and TODO updates:
  * README now reflects full array element support and notes remaining limitations (no direct printing of array elements, no loops yet).
  * TODO updated with future array enhancements (runtime bounds checking, richer index expressions, element printing).

* Added test case `error_intarray_bad_constaccess.silly` (currently commented in testerrors.sh – static bounds checking catches the error at compile time).

---

<!-- vim: set tw=80 ts=2 sw=2 et: -->
