# Silly Compiler Code Review & TODO List

## Updated: March 1, 2026

Review covers changes since commit `babc50cc` (Feb 16, 2026) — 58 commits.

---

## 📊 PROJECT STATUS OVERVIEW

### Metrics

| Metric | Feb 16 | Mar 1 | Change |
|--------|--------|-------|--------|
| Source lines (src/driver/) | ~6,200 | ~7,379 | +1,179 |
| `.silly` test files | ~190 | 223 (endtoend) + 22 (lit) | +55 |
| `TODO/FIXME/HACK` markers | ~10 | ~38 | ↑ (mostly "coverage" stubs) |
| Driver source files | 14 | 16 | +2 |

The `TODO: coverage` explosion in `CompilationUnit.cpp` and `SourceManager.cpp` is not a regression — it reflects error paths that are genuinely hard to trigger in normal tests, and marking them is the right practice. They are a testing debt line item, not a code quality problem.

---

## ✅ COMPLETED SINCE LAST REVIEW (Feb 16 → Mar 1)

### 1. MODULE / IMPORT System ✅ 🎉

**This was listed as "2-3 weeks" of future work in the last review. It's done.**

- `MODULE;` declaration supported in grammar and parser.
- `IMPORT foo;` statement compiles and works end-to-end.
- Modules pre-compiled to `.mlirbc` via `--emit-mlirbc -c`, then passed to the importer via `--imports file`.
- `ParseListener` walks imported `mlir::ModuleOp` and clones `FuncOp` prototypes into the importing module, with deduplication (`IMPORT mod2` twice in `call2modules.silly` is handled cleanly).
- LIT tests cover: single module, two modules, a module imported by two sources, bad module error.
- `SourceManager::findMOD()` handles the module lookup.
- Design notes committed to `Docs/module_import_design.md`.

**Quality:** ⭐⭐⭐⭐⭐ Major feature, working end-to-end.

---

### 2. Function Prototypes ✅

- `FUNCTION bar ( INT16 w );` (declaration without body) is now valid syntax.
- Test cases: `proto1.silly`, `proto2.silly`, `proto3.silly` cover declaration-only, declaration + definition, and declaration + call + definition ordering.
- `isExternal()` check in the lowering context skips prototype FuncOps correctly.

**Quality:** ⭐⭐⭐⭐⭐ Clean implementation.

---

### 3. LLVM-IR Round-trip (`.ll` / `.bc` import) ✅

- `--emit-llvmbc` flag added, emitting LLVM bitcode.
- `.ll` and `.bc` files accepted as input sources (round-trip: compile to `.ll`/`.bc`, then use as input).
- `InputType::LLVMIR` path added to `CompilationUnit`.
- `parseIRFile` used for auto-detection of text vs binary LLVM IR.
- README updated with usage examples.

**Quality:** ⭐⭐⭐⭐ Solid implementation.

---

### 4. `--emit-mlirbc` ✅

- MLIR bytecode emission implemented.
- `--emit-mlir` and `--emit-mlirbc` are mutually exclusive (enforced with error message).
- `--emit-llvm` and `--emit-llvmbc` are similarly mutually exclusive.
- LIT tests for all four emit modes (`emit-mlir-o`, `emit-mlirbc-o`, `emit-llvm-o`, `emit-llvmbc-o`, and their `outputdir` variants), generated via `generate-lit-emit-tests.pl`.

**Quality:** ⭐⭐⭐⭐⭐ Well-tested.

---

### 5. `--silly-version` Flag ✅

- `silly --silly-version` prints version string and LLVM path.
- Version bumped to `V0.10.0`.
- LIT test covers the output format.
- No longer requires a dummy input file (see item 10 below).

**Quality:** ⭐⭐⭐⭐⭐

---

### 6. `-S` Flag ✅

- Emit assembly (`.s`) without linking.
- `-S` and `-c` together correctly rejected with an error.

---

### 7. `fatalDriverError` Eliminated ✅

- `fatalDriverError()` removed from `CompilationUnit` and `SourceManager`.
- All error paths now return `ReturnCodes` values and unwind cleanly through the call stack.
- Destructors can run properly — no more `exit()` skipping cleanup.

**Quality:** ⭐⭐⭐⭐⭐ Significant correctness and RAII improvement.

---

### 8. `SourceManager` Introduced ✅

- New class orchestrates multi-source compilation.
- Owns the set of `CompilationUnit`s, indexed by filename stem.
- Manages: output directory construction, MLIR/LLVM serialization, object file collection, linking.
- `CompilationUnit` now takes a `SourceManager&` reference instead of owning driver state directly.
- Filename handling (stem, output path construction) moved out of `CompilationUnit` into `SourceManager`.

**Quality:** ⭐⭐⭐⭐ Good separation. See notes below about raw `new` usage.

---

### 9. LIT Test Infrastructure ✅

- `tests/lit/driver/` with `lit.cfg.py`, `FileCheck`-based tests.
- Covers: driver flags, emit modes, module imports, error cases, version flag.
- `tests/lit/dialect/` for MLIR dialect verification tests.
- `generate-lit-emit-tests.pl` script auto-generates the emit flag LIT tests.
- `cmake/RunFailureTest.cmake` and `cmake/RunTest.cmake` for CTest integration.

**Quality:** ⭐⭐⭐⭐⭐ LIT is the right tool for compiler driver tests.

---

### 10. `--silly-version` Without Dummy Input File ✅

- `silly --silly-version` no longer requires a dummy source file argument.
- The version check short-circuits before the "no input files" validation.
- `tests/lit/driver/not-enough-args-should-fail.silly` added to cover the actual no-args error path.
- `silly-version.silly` LIT test updated accordingly.

**Quality:** ⭐⭐⭐⭐⭐ Clean fix, properly tested.

---

### 11. `-o` with `--output-directory` Fixed ✅

- `-o exename` combined with `--output-directory` now works correctly.
- `--keep-temp` flag added and tested.
- LIT test covers the `-o` + `--output-directory` combination.

---

### 12. Multiple Source + `-c`/`-o` Fixes ✅

- Regression in `-c`/`-o` case from multi-source support fixed (object file wasn't being passed to linker).
- `-c` and `-o` used together now correctly handled.
- `multi-source-with-c-and-o-should-fail.silly` LIT test added.

---

### 13. Test Infrastructure: `add_endtoend_compile_tests` Flag Fix ✅

- The per-test flag override (`COMPILER_FLAGS_${test_name}`) was replacing instead of appending defaults.
- Fixed to `list(APPEND ...)`, so the 4 always-recompiling tests (`fatal`, `arrayprod`, `initlist`, `initarray`) are now stable.

---

### 14. LLVM 22.x Migration ✅

- Updated from LLVM/MLIR 21.x to 22.1.0+.
- Old `llvm21.flang.patch` removed.

---

## 🎯 HIGH PRIORITY (Do These Next)

### 1. ⭐⭐⭐ Raw `new` in `SourceManager::createCU`

**Current code:**
```cpp
CUs[stem] = FileNameAndCU( filename, new silly::CompilationUnit( *this ) );
```

**Problem:** `CompilationUnit` is heap-allocated with raw `new` and stored as a raw pointer in `FileNameAndCU`. The destructor in `SourceManager` presumably deletes them, but this is fragile and not idiomatic modern C++.

**Fix:** Use `std::unique_ptr`:

```cpp
struct FileNameAndCU
{
    std::string filename{};
    std::unique_ptr<silly::CompilationUnit> pCU{};
};

// In createCU:
CUs[stem] = FileNameAndCU{ filename, std::make_unique<silly::CompilationUnit>( *this ) };
```

Callers that receive `FileNameAndCU*` and access `pCU` would use `pCU.get()` where a raw pointer is needed.

**Estimated Effort:** 1-2 hours.

**Priority:** HIGH — correctness/safety.

---

### 2. ⭐⭐⭐ MODULE IMPORT: Remaining Design Gaps

The current IMPORT implementation is functional but has known gaps per the design doc and `TODO.md`:

**A) No timestamp-based recompilation**
Currently modules must be manually pre-compiled and passed via `--imports`. Automatic rebuild-if-stale is not implemented. The design doc covers the approach (`llvm::sys::fs::getLastModificationTime()`).

**B) No cycle detection**
The two-state DFS visited map described in the design notes is not yet implemented. A `A → B → A` cycle would currently either loop or crash.

**C) Name conflict behavior is undefined**
`README` notes: "Name conflicts between imported modules are undefined." The `lookupSymbol` deduplication guard handles the same module being imported twice, but two different modules defining the same symbol name is not caught.

**D) Signature mismatch between prototype and definition not checked**
`TODO.md` notes: "Consider an error if prototype and definition have mismatched signatures — currently probably crashes or silent mismatch; add sema check later (right now only check number of parameters, not return, nor types.)"

**Estimated Effort:** A) 1 day, B) half day, C) half day, D) 1 day.

**Priority:** HIGH — correctness gaps in the new feature.

---

### 3. ⭐⭐ Semantic Analysis Pass

Still the most impactful missing piece. Unchanged from last review — still not started. Key items:

- Non-const initializers (`error_nonconst_init.silly`) — currently not caught.
- `error_intarray_bad_constaccess.silly` — should fail but doesn't.
- Return path checking — `function_without_return.silly` is in the failure test suite but catching it earlier in a sema pass would be cleaner.
- Prototype/definition signature mismatch (see above).
- `CALL` usage: a function with a return type used as a standalone statement (no assignment).

**Estimated Effort:** 4-5 days.

**Priority:** HIGH — catches entire classes of errors.

---

### 4. ⭐⭐ `TODO: coverage` in `CompilationUnit.cpp` and `SourceManager.cpp`

There are ~20 error paths marked `// TODO: coverage` — these are real error conditions (bad filenames, failed opens, etc.) that don't yet have test cases driving them. They should be exercised.

For `CompilationUnit.cpp`, most of these are in the LLVM lowering and codegen paths. For `SourceManager.cpp`, they include directory creation failures, duplicate CU detection, and linker errors.

The LIT framework is already in place — these are good candidates for `// RUN: %Not %ExeSilly ...` tests.

**Estimated Effort:** 2-3 days.

**Priority:** MEDIUM-HIGH — coverage gaps in new infrastructure.

---

## 🔧 MEDIUM PRIORITY

### 5. `std::format` → `llvm::formatv`

Multiple `FIXME: probably want llvm::formatv` comments remain, particularly in `CompilationUnit.cpp` and `SourceManager.cpp`. The `std::string(path)` casting hack is present in several error messages. `llvm::formatv` handles `StringRef` and `SmallString` natively.

**Estimated Effort:** 2-3 hours, mechanical search-and-replace.

**Priority:** MEDIUM — code cleanliness.

---

### 6. Debug Info Issues (Unchanged from Last Review)

Still open, still impactful. Three distinct sub-problems with different root causes and fixes.

#### A) Induction Variable Scoped at Function Level

**Problem:** Loop induction variables show function-level scope in the debugger, instead of being scoped to the loop body's lexical block. This will behave strangely if the same variable name is used in multiple loops.

**Current (wrong) DWARF:**
```
DW_TAG_variable
  DW_AT_name        : i
  DW_AT_decl_file   : for_simplest.silly
  DW_AT_decl_line   : 3
  DW_AT_type        : int64_t
  DW_AT_location    : (function scope)  // WRONG
```

**Desired DWARF:**
```
DW_TAG_lexical_block
  DW_AT_low_pc      : <FOR loop start>
  DW_AT_high_pc     : <FOR loop end>

  DW_TAG_variable
    DW_AT_name      : i
    DW_AT_decl_line : 3
    DW_AT_type      : int64_t
    DW_AT_location  : <block scope>  // CORRECT
```

**Fix approach:** In `ParseListener.cpp`, create a `DILexicalBlockOp` when entering the FOR loop and push/pop a DI scope stack:

```cpp
void ParseListener::enterForStatement( SillyParser::ForStatementContext* ctx )
{
    auto forBodyBlock = state.builder.create<mlir::LLVM::DILexicalBlockOp>(
        loc,
        state.diCompileUnit,   // parent scope
        state.diFile,
        ctx->getStart()->getLine(),
        ctx->getStart()->getCharPositionInLine()
    );
    perFunctionState.pushDIScope( forBodyBlock );
    // ... create induction variable DI using perFunctionState.currentDIScope() ...
}

void ParseListener::exitForStatement( SillyParser::ForStatementContext* ctx )
{
    perFunctionState.popDIScope();
}
```

**Verify with:**
```bash
silly -g for_simplest.silly
llvm-dwarfdump --debug-info for_simplest | grep -A 10 "DW_AT_name.*\"i\""
# Should show DW_TAG_lexical_block as parent, not subprogram
```

**Estimated Effort:** 2 days.

---

#### B) Line Number Ping-Pong in Loop Bodies

**Problem:** Stepping through a loop in gdb alternates unexpectedly between statements. Example from `printdi.silly`:

```
(gdb) b 6
Breakpoint 2 at 0x400798: file printdi.silly, line 6.
(gdb) run
Breakpoint 2, main () at printdi.silly:6
6           t = c[i];
(gdb) n
8           PRINT "c[", i, "] = ", t;
(gdb) n
6           t = c[i];    // <-- jumped back unexpectedly
(gdb) n
8           PRINT "c[", i, "] = ", t;
```

**Root cause:** Intermediate IR ops (GEPs, loads for array accesses, loop increment/condition) are inheriting or emitting location info from the wrong source location. The `unknownLoc` suppression for PRINT intermediates was partially applied but didn't fully fix the issue in `arrayprod.silly`.

**Investigation approach:**
1. Dump the LLVM IR with `-g --emit-llvm` and inspect `!dbg` annotations on all ops in the loop body.
2. Look for any `!dbg` referencing the wrong line on GEP, load, or store instructions that should be attributed to no location or an earlier line.
3. The `initFill` memset `HACK: suppress location info` in `LoweringContext.cpp` is a related symptom — the same pattern of re-ordering causing spurious location attribution may be at play elsewhere.

**Estimated Effort:** 2-3 days (investigation-heavy).

---

#### C) Multi-Argument PRINT Line Stepping

**Problem:** Each argument to a multi-argument PRINT generates a separate line entry, causing gdb to stop multiple times per PRINT statement. Example from `arrayprod.silly`:

```
(gdb) n
34          PRINT "c[", i, "] = ", c[i], " ( = ", v, " * ", w, " )";
(gdb)
32          c[i] = v * w;    // <-- wrong, jumped back
(gdb)
34          PRINT ...
(gdb)
32          c[i] = v * w;
```

**Root cause:** The multiple runtime calls emitted for a multi-argument PRINT each carry their own `!dbg` location pointing back to the PRINT source line, and the GEPs and loads for each argument carry locations too. LLVM's line table generation sees these as distinct line entries.

**Fix approach:** Use `mlir::FusedLoc` to group all ops that contribute to a single PRINT into one logical location, and suppress intermediate op locations:

```cpp
// In lowering, when lowering a PrintOp with multiple args:
llvm::SmallVector<mlir::Location> argLocs;
for ( auto& arg : printArgs )
    argLocs.push_back( arg.getLoc() );

auto fusedLoc = mlir::FusedLoc::get( context, argLocs );

// All intermediate GEPs, loads, bitcasts get unknownLoc:
auto loadOp = rewriter.create<mlir::LLVM::LoadOp>( rewriter.getUnknownLoc(), ... );

// Only the final print call gets the real source location:
auto callOp = rewriter.create<mlir::LLVM::CallOp>( fusedLoc, ... );
```

The key insight is that `FusedLoc` is not just a multi-location — it signals to LLVM's DI emission that all the fused locations belong to the same logical instruction, suppressing intermediate line table entries.

**Test:** `print_multiple.silly` — stepping should stop exactly once per PRINT statement.

**Estimated Effort:** 1-2 days.

---

#### D) Line Stepping After CALL Returns

**Problem:** After returning from a function call, the debugger shows the wrong line. From `function.silly`:

```
(gdb) n
main () at function.silly:39
39      PRINT v;                    // OKAY
(gdb) n
38      v = CALL add( 42, 0 );      // WRONG — already done
(gdb) s
39      PRINT v;                    // WRONG — already done
```

**Root cause:** Suspected to be the same pattern — ops after the call site are picking up the call site's location, causing gdb to re-visit it. Worth investigating whether `func.call` is emitting a location on the return-value extraction, and whether the ops immediately following the call need `unknownLoc` or a fresh location.

**Estimated Effort:** 1-2 days.

---

#### E) Automated DWARF Tests

Manual `llvm-dwarfdump` testing is the current practice. This should be automated to prevent regressions as the debug info work progresses. A LIT-based approach using `llvm-dwarfdump` + `FileCheck` is cleaner than a Python test framework:

```
// tests/lit/debug/for_induction_var.silly
// RUN: %ExeSilly -g %s -o %t
// RUN: llvm-dwarfdump --debug-info %t | %FileCheck %s
//
// CHECK: DW_TAG_lexical_block
// CHECK-NEXT: DW_AT_decl_file
// CHECK: DW_TAG_variable
// CHECK-NEXT: DW_AT_name ("i")
// CHECK-NEXT: DW_AT_decl_file
// CHECK-NEXT: DW_AT_decl_line (3)
```

This integrates naturally into the existing LIT infrastructure and doesn't require any new test framework.

**Estimated Effort:** 1 day to set up the pattern; then incremental as debug fixes are made.

**Total Effort for All Debug Info Work:** 7-10 days.

**Priority:** MEDIUM — affects developer experience significantly, but not compiler correctness.

---

### 7. `SourceManager.cpp` LLVM Path Separator FIXME

```cpp
// FIXME: there is portable LLVM infra for path construction (but I don't currently care about Windows, so "/"
```

This hardcodes `/` as the path separator. Use `llvm::sys::path::append()` consistently.

**Estimated Effort:** 30 minutes.

**Priority:** MEDIUM — low risk, easy fix.

---

### 8. Error Message Polish (Unchanged from Last Review)

- Error numbers/classes (`E0042` style) — not started.
- Error count limit (`-ferror-limit`) — not started.

**Estimated Effort:** 2-3 days.

**Priority:** MEDIUM.

---

### 9. `showLinkCommand` Coverage

```cpp
static void showLinkCommand( ... )
{
    // TODO: coverage
```

`--verbose-link` flag exists but `showLinkCommand` has no test. Add a LIT test that passes `--verbose-link` and checks stderr for the linker invocation.

**Estimated Effort:** 30 minutes.

**Priority:** LOW-MEDIUM.

---

## 🔨 STILL OPEN FROM PREVIOUS REVIEW

These items from the Feb 16 review remain unaddressed:

| Item | Status | Notes |
|------|--------|-------|
| BREAK/CONTINUE | Not started | `TODO.md` still lists |
| WHILE loops | Not started | `TODO.md` still lists |
| Runtime bounds checking | Not started | `ABORT` builtin now available as foundation |
| `error_intarray_bad_constaccess.silly` bug | Not fixed | Listed in Bugs section of `TODO.md` |
| `modfloat.silly` broken test | Not fixed | Listed in `TODO.md` |
| Hardcoded `/build/` paths | Not fixed | `TODO.md` lists `bin/build`, `tests/dialect/lit.cfg.py` |
| `lowering.cpp` cut-and-paste type conversion | Not fixed | |
| DWARF automated tests | Not started | |
| Type converter integration (DeclareOp) | Not started | Tracked as "Attempted, reverted" |
| `--ferror-limit` | Not started | |
| Error numbers | Not started | |

---

## 🚀 NEW ITEMS IDENTIFIED IN THIS REVIEW

### A) `--imports` UX: Explicit vs Automatic

Currently the user must manually pre-compile modules and pass them with `--imports file`. This is workable but clunky for multi-module projects. The design doc describes automatic build-on-demand. Even without full automatic recompilation, a convention like "look for `foo.mlirbc` in the same directory as the source" would reduce friction.

### B) `SourceManager` Owns Too Much State

`SourceManager` currently holds `outdir`, `exeName`, `defaultExecutablePath`, `objFiles`, `tmpToDelete`, and orchestrates linking. As the driver grows (import resolution, topological sort), this class may become unwieldy. Consider whether a `Driver` or `CompilationDriver` wrapper class should own `SourceManager` and orchestrate the compilation pipeline at a higher level — this is already noted in `TODO.md`.

### C) `generate-lit-emit-tests.pl`

The Perl script that generates LIT tests for emit modes is a clever idea, but it's an undocumented build tool. Add a comment at the top of the generated test files noting they are auto-generated and how to regenerate, and add a `make` or `cmake` target to regenerate them.

### D) `tests/endtoend/sir/` Commented Out

`tests/endtoend/CMakeLists.txt` has:
```cmake
# haven't generated expected outputs for this one -- am not sure I should:
#add_subdirectory(sir)
```

The `.sir` files in that directory are valid test inputs. Either add expected outputs and enable the subdirectory, or remove the directory to avoid confusion.

---

### 10. Runtime Bounds Checking

**Why:** Safety — better error messages than segfaults. The `ABORT` builtin is already in place as a foundation.

**What:** Add `--bounds-check` compiler flag for optional array access validation.

**Driver:**
```cpp
static llvm::cl::opt<bool> enableBoundsCheck(
    "bounds-check",
    llvm::cl::desc( "Enable runtime array bounds checking" ),
    llvm::cl::init( false ),
    llvm::cl::cat( SillyCategory )
);
```

**Lowering** (in array LoadOp/AssignOp lowering, when `ds.boundsCheck` is set):
```cpp
mlir::Value arraySize = rewriter.create<mlir::arith::ConstantIndexOp>( loc, arrayType.getSize() );

mlir::Value negative = rewriter.create<mlir::arith::CmpIOp>(
    loc, mlir::arith::CmpIPredicate::slt, index, zeroIndex );
mlir::Value tooLarge = rewriter.create<mlir::arith::CmpIOp>(
    loc, mlir::arith::CmpIPredicate::sge, index, arraySize );
mlir::Value outOfBounds = rewriter.create<mlir::arith::OrIOp>( loc, negative, tooLarge );

auto ifOp = rewriter.create<mlir::scf::IfOp>( loc, outOfBounds, /*withElseRegion=*/true );

// Then block: call runtime abort
rewriter.setInsertionPointToStart( ifOp.thenBlock() );
rewriter.create<mlir::func::CallOp>( loc, abortOobFunc,
    mlir::ValueRange{ filenameCst, lineCst, index, arraySize } );
rewriter.create<mlir::func::ReturnOp>( loc );   // unreachable

// Else block: normal load
rewriter.setInsertionPointToStart( ifOp.elseBlock() );
mlir::Value loaded = rewriter.create<mlir::LLVM::LoadOp>( loc, elemPtr );
rewriter.create<mlir::scf::YieldOp>( loc, loaded );

rewriter.replaceOp( op, ifOp.getResults() );
```

**Runtime addition** (`src/runtime/Silly_runtime.cpp`):
```cpp
extern "C" [[noreturn]] void __silly_abort_oob(
    const char* file, int line, int64_t index, int64_t size )
{
    fprintf( stderr, "%s:%d: runtime error: array index %ld out of bounds [0, %ld)\n",
             file, line, index, size );
    abort();
}
```

**Usage:**
```bash
silly --bounds-check myprogram.silly
./myprogram
# myprogram.silly:42:5: runtime error: array index 10 out of bounds [0, 5)
```

Zero overhead when not enabled; minimal (single branch per access) when enabled.

**Estimated Effort:** 3-4 days.

**Priority:** MEDIUM — good safety feature, foundation already in place.

---

### 11. Output File Cleanup on Error

**Why:** If compilation fails partway through, partially-written output files are left on disk. LLVM provides `TempFile` for this:

```cpp
// Instead of opening raw output file:
auto tempOrErr = llvm::sys::fs::TempFile::create( "silly-obj-%%%%%%.o" );
if ( !tempOrErr ) { /* handle error */ }

// On success, keep the file:
if ( auto err = tempOrErr->keep( finalPath ) ) { /* handle */ }

// On failure, discard automatically via TempFile destructor
```

For the `--output-directory` case this needs care — `TempFile::create` uses a fixed prefix pattern, so you'd create in `$TMPDIR` and rename on success. The existing `tmpToDelete` vector in `SourceManager` is the current manual approach; replacing it with `TempFile` RAII would be cleaner.

**Estimated Effort:** 2-3 hours.

**Priority:** LOW — polish, not correctness.

---

### 12. Type Converter Integration for DeclareOp (Low Priority / Exploratory)

**Background:** A previous attempt to use `mlir::TypeConverter` to automatically remap `!silly.var<T>` → `!llvm.ptr` (and update all uses of `DeclareOp`) didn't work and was reverted to an `unordered_map<Operation*, Operation*>`. The map approach works and is not broken. This is exploratory cleanup, not a correctness issue.

**Why the type converter approach failed:**
- All patterns must use `ConversionPatternRewriter`, not `PatternRewriter`.
- `materializeSourceConversion` callbacks may be needed for types that escape the conversion.
- The converter must be consistently applied — partial application causes the rewriter to fail to legalize.

**Sketch of the proper approach:**
```cpp
class SillyToLLVMTypeConverter : public mlir::TypeConverter
{
public:
    SillyToLLVMTypeConverter()
    {
        addConversion( []( silly::VarType type ) -> mlir::Type
        {
            return mlir::LLVM::LLVMPointerType::get( type.getContext() );
        } );
        addConversion( []( silly::ArrayType type ) -> mlir::Type
        {
            return mlir::LLVM::LLVMArrayType::get(
                convertType( type.getElementType() ), type.getSize() );
        } );
        addConversion( []( mlir::Type type ) { return type; } ); // passthrough
    }
};
```

All `OpConversionPattern` subclasses must then use the converter-aware `matchAndRewrite(Op, OpAdaptor, ConversionPatternRewriter&)` signature, and `populateConversionTarget` must mark all silly types as illegal post-conversion.

**Estimated Effort:** 2-3 days (risky, exploratory).

**Priority:** LOW — current `unordered_map` approach works fine. Only worth attempting if the map causes maintenance problems.

---

## 📋 QUICK WINS (< 1 Hour Each)

1. **`std::format` → `llvm::formatv`** (1 hour) — mechanical search-and-replace, handles `StringRef`/`SmallString` natively without the `std::string(path)` casting hack.
2. **Add `--verbose-link` LIT test** (30 min) — removes a `// TODO: coverage` in `showLinkCommand`.
3. **Fix LLVM path separator FIXME in `SourceManager.cpp`** (30 min) — use `llvm::sys::path::append` instead of hardcoded `/`.
4. **Comment generated LIT tests** (15 min) — add a note at the top of emit tests saying they are generated by `generate-lit-emit-tests.pl` and how to regenerate.
5. **Resolve `tests/endtoend/sir/`** (30 min) — either add expected outputs and `add_subdirectory(sir)`, or remove the directory. Currently disabled with a comment.
6. **`--emit-mlir` dependency in `add_endtoend_compile_tests`** (1 hour) — `TODO.md` notes the `--emit-llvm --emit-mlir` defaults are a holdover from `testit`. Consider whether they are actually needed for the CTest run tests, or just for manual inspection.

---

## 🗓️ RECOMMENDED NEXT STEPS

**Immediate (this week):**
1. Fix raw `new` in `SourceManager::createCU` (1-2 hours).
2. Add cycle detection to IMPORT (half day).
3. Add duplicate symbol name detection to IMPORT (half day).

**Short term (2-3 weeks):**
4. `TODO: coverage` tests for `CompilationUnit` and `SourceManager` error paths.
5. Prototype/definition signature mismatch sema check.
6. Start semantic analysis pass — initializer checking is the easiest entry point.
7. Quick wins: `llvm::formatv` cleanup, `--verbose-link` LIT test, path separator fix.

**Medium term (1-2 months):**
8. Debug info: tackle the `FusedLoc` fix for PRINT line stepping (C above) — highest ROI of the debug items.
9. Debug info: induction variable lexical block scope (A above).
10. BREAK/CONTINUE.
11. Runtime bounds checking (foundation is in place with `ABORT`).

---

## 💡 OBSERVATIONS

### What's Going Really Well ✅

The MODULE/IMPORT implementation is the headline win — a feature estimated at 2-3 weeks of work was delivered in this period alongside significant other changes (LIT infrastructure, LLVM 22 migration, round-trip IR support, version flag, `-S` flag, multiple bug fixes). The architecture is holding up well under the expansion.

The LIT test infrastructure is the right long-term direction for driver-level tests. The `FileCheck`-based approach scales much better than the old `testit` script.

The `fatalDriverError` elimination is a meaningful correctness improvement — RAII and destructor semantics now work as expected throughout the compilation pipeline.

### Areas of Concern 📈

The `TODO: coverage` count in `CompilationUnit.cpp` is high (14 markers). This isn't blocking anything, but it means a large fraction of error-handling code is untested. These paths tend to be exactly the ones that matter when something goes wrong in the field.

The IMPORT system is functional but the known gaps (no cycle detection, undefined name conflict behavior, no signature mismatch check) should be addressed before the feature is considered complete. The design doc already has the right answers — it's a matter of implementing them.

### What Not to Do ⚠️

Don't start WHILE loops or multi-dimensional arrays until the IMPORT correctness gaps are closed and the sema pass has begun. The module system is now a core feature and its correctness matters more than new syntax.

---

*Review based on commit range `babc50cc..HEAD` (58 commits, Feb 16 → Mar 1, 2026).*

<!-- vim: set tw=100 ts=2 sw=2 et: -->
