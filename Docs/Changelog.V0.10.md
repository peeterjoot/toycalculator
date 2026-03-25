# Changelog V0.10.0

## LLVM Version

Updated to llvm-project 22.1.0. V9 was the last version supporting llvm-project 21.x.
Corresponding changes to `builder.create<>` syntax and related MLIR API updates throughout.

---

## Major Features

### MODULE / IMPORT System

A working multi-file compilation system. The `MODULE` and `IMPORT` keywords are now fully
supported end-to-end.

**How it works:**

- A source file beginning with `MODULE;` declares a module. It may contain only `FUNCTION`
  and `IMPORT` statements.
- Other sources use `IMPORT name;` to bring a module's functions into scope as prototypes.
- The compiler is invoked with `--imports module.silly` to name the module source.

**Processing algorithm:**

1. The `--imports` module is compiled as far as MLIR (can also be a `.mlirbc` or `.mlir`
   file; cannot be `.ll` or `.o`).
2. The remaining silly sources are compiled to object code. Any `IMPORT` statements are
   processed by cloning `FuncOp` declarations from the module into the importing compilation
   unit.
3. The `--imports` module is lowered to object code.
4. Everything is linked.

**Demo:**

```
fedoravm:~> cat callee.silly
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

fedoravm:~> cat callmod.silly
IMPORT callee;

PRINT "In main";
CALL bar( 3 );
PRINT "Back in main";

INT32 rc = CALL foo();
PRINT "Back in main again, rc = ", rc;

fedoravm:~> silly --imports callee.silly callmod.silly && ./callmod
In main
3
Back in main
3
Back in main again, rc = 42
```

Additional capabilities:

- `IMPORT` is allowed within a `MODULE` (a module can import another module's functions).
- Multiple source files can all `IMPORT` the same module — deduplication is handled
  automatically.
- Repeated `IMPORT` of the same module within one file is allowed and idempotent.
- `--imports` can be combined with `-c` for the submodule.
- Error is emitted (not assert) when the named module cannot be found.
- LIT tests cover: single module, two modules, module imported by two sources, bad module,
  repeated import, explicit `MAIN`.

**Design notes:** `Docs/module_import_design.md`.

---

### Function Prototypes

`FUNCTION name( params ) : returntype;` (declaration without body) is now valid syntax.
Allows forward declarations and separation of interface from implementation.

Test cases: `proto1.silly`, `proto2.silly`, `proto3.silly` — cover declaration-only,
declaration + definition, and declaration + call + definition ordering.

---

### LLVM-IR Round-Trip (`.ll` / `.bc`)

- `--emit-llvmbc` flag added, emitting LLVM bitcode (`.bc`).
- `.ll` (LLVM IR text) and `.bc` (LLVM bitcode) files now accepted as input sources.
- `parseIRFile` used for auto-detection of text vs binary LLVM IR.
- `--stdout` option removed (superseded by `--debug`).
- LLVM IR is now dumped only after optimization passes have run.
- `buildPerModuleDefaultPipeline` omitted at `-O0` to avoid spurious pass overhead.

---

### Debug Information: Lexical Block Scopes

- `silly::DebugScopeOp` introduced in the dialect. `DebugNameOp` now takes an optional
  scope as an input.
- Per-function stack of `DebugScopeOp`s maintained in `ParserPerFunctionState`.
- Lowering constructs `DILexicalBlockAttr` for each `DebugScopeOp`.
- `DILexicalBlockAttr` enabled for `IF`, `ELSE`, `ELIF` blocks, but not `FOR` loop bodies.
- Variables declared inside control flow blocks are now correctly scoped to their block
  in DWARF output — they no longer leak to function scope.
- `DeclareOp` insertion point no longer moved to function entry; declarations are emitted
  in-place at their correct location in the IR.
- FOR induction variables intentionally excluded from lexical block scoping for now (DI
  for induction variables in loop scope is a known open item).

New tests: `if-scope.silly`, `induction-var-simple.silly`, `different-scoped-vars.silly`,
`induction-var-and-scope-decl.silly`.

---

### DWARF5

Switched from DWARF4 to DWARF5. `targetMachine` is now set in `llvmModule` before dumping
(previously set just before the assembly printer — ordering was fragile). All debug tests
updated for `dwarfdump` output format differences between DWARF4 and DWARF5.

---

### Experimental Bison/Flex Frontend

An alternative parser frontend using Bison and Flex, targeting environments where
ANTLR4 is not available (e.g. LLVM builds without RTTI). Build with
`-DUSE_BISON_GRAMMAR=ON`. Automatically selected by `bin/build` when the LLVM build
has `-fno-rtti`.

**Status at HEAD: 174/248 tests passing.**

Grammar (`src/bisonGrammar/silly.y`) supports:

- All scalar types: `INT8`, `INT16`, `INT32`, `INT64`, `FLOAT32`, `FLOAT64`, `BOOL`
- Array declarations with bounds and initializer lists (including empty `{}`)
- Full expression hierarchy with correct operator precedence and associativity
  (`%left`/`%right`/`%nonassoc` Bison declarations)
- All binary operators: arithmetic, logical, bitwise, comparison
- Unary operators: negation, `+`, `NOT`
- Parenthesized expressions
- `PRINT` with expression lists, `CONTINUE` modifier, `ERROR` variant
- `ASSIGN` (scalar and array element)
- `CALL` as both statement and expression
- `FUNCTION` — prototypes and full definitions with parameter lists and return types
- `IF` / `ELIF` / `ELSE` with scoped statement blocks
- `FOR` with 2-argument and 3-argument range
- `GET`, `ABORT`, `IMPORT`, `EXIT`, `RETURN`, `MODULE`, `MAIN`
- Proper Flex location tracking via `YY_USER_ACTION`
- Float and string literal lexing (combined with integer in one Flex rule to avoid
  ambiguity)
- Range checking for integer literals in the Flex rule

Walker (`src/bisonGrammar/BisonParseListener.cpp`) delegates all MLIR construction
to the shared `Builder` base class.

---

### Builder: Grammar-Agnostic MLIR Construction

`src/driver/Builder.cpp` and `src/include/Builder.hpp` introduced as a shared base
class for both `Antlr4ParseListener` and `BisonParseListener`. Contains all
grammar-agnostic MLIR builder logic. Both parse frontends inherit from `Builder`.

Methods factored from `Antlr4ParseListener` / `ParseListener` into `Builder`:

- Literal creation: `createBooleanFromString`, `createIntegerFromString`,
  `createFloatFromString`, `createStringLiteral`
- Variable management: `createDeclaration`, `createVariableLoad`, `lookupDeclareForVar`
- Assignment and casting: `createAssignment`, `createCastIfNeeded`, `createIndexCast`
- Control flow: `createReturn`, `getReturnType`
- Arithmetic: `createBinaryArith`, `createBinaryCompare`, `createUnary`
- Statements: `createGet`, `createImport`
- Functions: `createFunction`, `finishFunction`, `createCall`
- Loops: `createFor`, `finishFor`
- Conditionals: `createIf`, `selectElseBlock`, `finishIfElifElse`
- Scoping: `enterScopedRegion`, `exitScopedRegion`

The old monolithic `src/driver/ParseListener.cpp` (2,369 lines) has been replaced by
`src/antlr4Grammar/Antlr4ParseListener.cpp` (1,506 lines) and `src/driver/Builder.cpp`
(1,100 lines).

---

## Driver Changes

### New / Changed Flags

- `--silly-version` — prints compiler version and the LLVM/MLIR path used to build it.
  No longer requires a dummy input file argument.
- `--no-verbose-parse-error` — suppresses the source-line caret display in parse errors.
- `--emit-llvmbc` — emit LLVM bitcode (`.bc`).
- `--keep-temp` — keep temporary object files.
- `-S` option removed; use `--emit-llvm -c` instead.

### Fixes

- `-o exename` combined with `--output-directory` now works correctly.
- `-c` and `-o` used together now correctly handled; error message updated to:
  `error: -c and -o cannot be used together with multiple input files (ambiguous output name)`.
- `--emit-llvm` / `--emit-llvmbc` and `--emit-mlir` / `--emit-mlirbc` made mutually
  exclusive (enforced with error message and tests).
- `--emit-llvm*` / `--emit-mlir*` without `-c` fixed (was broken by earlier `-c`
  output-directory fixes).
- `fatalDriverError()` removed from `CompilationUnit` and `SourceManager`. All error
  paths now return `ReturnCodes` values and unwind cleanly through the call stack.
  Destructors now run correctly on all error paths.
- `silly --silly-version` no longer emits "not enough positional command line arguments".
- `SourceManager` raw `ModuleOp` usage reduced — prefer passing `OwningOpRef&` or
  keeping `ModuleOp` local.
- Version string changed from `V10` to `V0.10.0`; old changelog files renamed to match
  (`Changelog.V0.0.md`, `Changelog.V0.1.md`, etc.).

### `-fno-rtti` Support

- CMakeLists.txt detects `-fno-rtti` LLVM builds and automatically selects
  `-DUSE_BISON_GRAMMAR=ON`.
- `bin/build` detects no-RTTI LLVM and passes the cmake flag.
- `bin/buildllvm` has a new option for building `llvm-project` without RTTI.
- Driver sources that cannot coexist with `-fno-rtti` (due to ANTLR4 `dynamic_cast`
  usage) are isolated with per-target compile options in CMakeLists.

---

## Parser / Grammar Changes

### ANTLR4 Frontend

- Removed old `DCL_TOKEN` from grammar.
- Unified `=` assignment-style and `{}` uniform-initializer declaration syntax.
  Either form can now take arbitrary expressions (not just constants).
  The old restriction "initializer expressions must be constant expressions" is lifted.

### Bison/Flex Frontend

See "Experimental Bison/Flex Frontend" above.

---

## Code Quality

### `std::format` → `llvm::formatv`

Most `llvm::errs()` / `dbgs()` / `outs()` call sites switched from `std::format` to
`llvm::formatv`, which handles `StringRef` and `SmallString` natively without casting.

---

## Testing & Quality

### Test Infrastructure Consolidation

The test directory structure was completely restructured and simplified:

- `tests/endtoend/debug/` merged into `tests/lit/debug/`
- `tests/endtoend/driver/` merged into `tests/lit/driver/`
- `tests/endtoend/failure/` merged into `tests/lit/syntax/`
- `tests/endtoend/exit/`, `tests/endtoend/get/`, `tests/endtoend/fatal/` all migrated
  to `tests/`
- All remaining `tests/endtoend/` content moved to `Samples/` with auto-generated lit
  wrappers (via `bin/generate_lit_wrappers.pl`)
- `tests/lit/` hierarchy flattened into a single `tests/` directory with one
  `lit.cfg.py` and one `lit.site.cfg.py.in`
- `cmake/RunTest.cmake` and `cmake/RunFailureTest.cmake` deleted
- `FILECHECK_EXECUTABLE` removed from top-level CMakeLists (no longer needed)
- `--emit-mlir --emit-llvm` removed from `add_endtoend_compile_tests` (was a holdover)
- Systematic rename of all test files for clarity and consistency
- Generated LIT tests marked `// Auto-generated by generate-lit-emit-tests.pl`

### New Tests

- `--verbose-link` LIT test
- Driver error path coverage: `bad-file-should-fail`, `bad-llvm-ir`,
  `bad-mlir-output-path`, `bad-llvm-ir-output-path`, `bad-mlir-path`,
  `bad-mlir-should-fail`, `bad-object-output-path`, `bad-suffix-should-fail`
- `not-enough-args-should-fail.silly` — covers actual no-args error path
- `array-return-verbose.silly` — covers `--verbose-parse-error` codepath
- `emit-llvm-both-should-fail.silly`, `emit-mlir-both-should-fail.silly`
- `keep-temp-emit-binary.silly`, `keep-temp-emit-text.silly`
- Module import: `mod1.silly`, `mod2.silly`, `call2modules.silly`, `callmod.silly`,
  `callee.silly` — single module, two modules, repeated import, explicit MAIN
- Debug: `if-scope.silly`, `induction-var-simple.silly`, `different-scoped-vars.silly`,
  `induction-var-and-scope-decl.silly`, `two-declare-order-location.silly`
- `toolong-int-literal.silly`, `toolong-float-literal.silly` — range check for
  oversized literals (catching `std::exception`)
- `too-big-array.silly` — range check for oversized array index expression
- `proto1.silly`, `proto2.silly`, `proto3.silly` — function prototype tests
- `rc-3.silly` and related exit code tests fixed

### Ubuntu/WSL Port

Resurrected and updated to llvm-project 22.1.0 in the WSL2/Ubuntu environment.

---

## Known Issues & Limitations

- Bison frontend: 74/248 tests still failing at HEAD. Expression-level location
  info is propagated as the enclosing statement's start location — sub-expression
  error messages point to the statement rather than the expression.
- Bison frontend: `emitParseError` incorrectly routes through `emitInternalError`,
  making user syntax errors appear as internal compiler errors.
- MODULE IMPORT: no cycle detection, no cross-module symbol conflict detection, no
  prototype/definition signature mismatch check across module boundaries.
- FOR induction variables are emitted at function scope in DWARF, not loop scope.
- GDB line-stepping regressions in loop bodies and after CALL returns remain open.
- `Samples/expressions/modfloat.silly` — broken with mixed `FLOAT32`/`FLOAT64`.

---
