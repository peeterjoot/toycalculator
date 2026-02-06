# Silly Compiler TODO List
## Updated: February 8, 2026

This is a refreshed, prioritized TODO list based on the February 2, 2026 code review.

---

## ✅ Recently Completed (Since Last Review)

### Error Handling ✅
- **DONE:** Implemented `emitUserError()` with GCC/Clang-style formatting
- **DONE:** Source line context with column pointer (`^`)
- **DONE:** "In function 'foo':" headers for better context
- **DONE:** Internal vs user error differentiation
- **DONE:** Error counting and cascading error suppression

**Quality:** Production-level error reporting! 🎉

### Project Organization ✅
- **DONE:** Hierarchical test structure (`tests/endtoend/{array,bool,builtins,...}/`)
- **DONE:** 178 comprehensive test cases
- **DONE:** Clean source layout (`src/{driver,dialect,grammar,runtime,include}/`)
- **DONE:** Per-directory CMakeLists.txt files

### Build & Packaging ✅  
- **DONE:** CMake install rules with `SILLY_ENABLE_INSTALL` option
- **DONE:** Configurable `CMAKE_INSTALL_PREFIX`
- **DONE:** Installs binaries, libraries, docs, samples
- **DONE:** Proper FHS layout (`bin/`, `lib/`, `share/silly/`)

### Documentation ✅
- **DONE:** Doxygen configuration customized for project
- **DONE:** Output to `Docs/doxygen/`
- **DONE:** Call graphs, source browsing, TreeView enabled

---

## 🎯 HIGH PRIORITY (Do These First)

### 1. ⭐⭐⭐ Semantic Analysis Pass

**Why:** Catches errors earlier, improves error messages, enables better optimizations

**Status:** Not started (mentioned in TODO #33)

**What to Build:**
```cpp
// File: src/include/semantics.hpp
namespace silly {

class SemanticAnalyzer {
public:
    explicit SemanticAnalyzer(mlir::ModuleOp module) : module(module) {}
    
    /// Run all semantic checks, return failure if errors found
    mlir::LogicalResult analyze();
    
private:
    mlir::ModuleOp module;
    unsigned errorCount = 0;
    
    /// Check initializers are constant expressions (no variables)
    mlir::LogicalResult checkInitializers();
    
    /// Verify all function paths return a value (if return type non-void)
    mlir::LogicalResult checkReturnPaths();
    
    /// Validate array bounds where statically determinable
    mlir::LogicalResult checkArrayBounds();
    
    /// Verify no forward references in initializers
    mlir::LogicalResult checkDeclarationOrder();
    
    /// Check CALL used correctly (assignment if return type, standalone if void)
    mlir::LogicalResult checkCallUsage();
};

} // namespace silly
```

**Implementation Steps:**
1. Create `src/include/semantics.hpp` and `src/driver/semantics.cpp`
2. Implement each checker as a separate walk of the MLIR module
3. Integrate into `driver.cpp` between parse and lowering
4. Add tests for each error case

**Test Cases to Fix:**
- `error_nonconst_init.silly` - should catch non-const in initializer
- `error_intarray_bad_constaccess.silly` - should catch OOB access
- `array_elem_as_arg.silly` - better error for missing RETURN
- Forward reference detection in initializers

**Estimated Effort:** 3-4 days

**Priority:** HIGH - This is the missing piece in your compilation pipeline

**Integration Point:**
```cpp
// In driver.cpp, after parsing:
if (emitMLIR || emitLLVM) {
    silly::SemanticAnalyzer analyzer(theModule);
    if (mlir::failed(analyzer.analyze())) {
        llvm::errs() << "Semantic analysis failed\n";
        return 1;
    }
    
    // Continue to lowering...
}
```

---

### 2. ⭐⭐ Debug Info Quality Improvements

**Why:** Line-stepping issues frustrate debugging experience

**Current Issues (from TODO):**
- Loop bodies have line number ping-pong (TODO #41-72)
- Multi-argument PRINT creates multiple line entries (TODO #81-100)
- Induction variables scoped at function level instead of lexical block (TODO #13)

**Tasks:**

#### A) Fix Loop Variable Scope
**Problem:** Loop induction variables show function-level scope in debugger

**Solution:**
```cpp
// In parser.cpp, when entering FOR loop:
void ParseListener::enterForStatement(...) {
    // Create DILexicalBlock for loop body
    auto forBodyBlock = createLexicalBlock(loc, currentScope);
    
    // Attach induction variable DI to loop block, not function
    constructInductionVariableDI(inductionVar, forBodyBlock);
}
```

**Test:** Verify with `dwarfdump` on `for_simplest.silly`:
```bash
dwarfdump out/for_simplest | grep -A 5 "DW_AT_name.*i"
# Should show DW_AT_decl_line matching loop, not function start
```

**Estimated Effort:** 1 day

#### B) Consolidate PRINT Location Info
**Problem:** Multi-argument print shows separate locations for each arg

**Current (Bad):**
```
(gdb) n
34          PRINT "c[", i, "] = ", c[i], " ( = ", v, " * ", w, " )";
(gdb)
32          c[i] = v * w;    // <-- jumped back!
(gdb)
34          PRINT "c[", i, "] = ", c[i], " ( = ", v, " * ", w, " )";
```

**Solution:**
```cpp
// In lowering, when creating print operations:
auto fusedLoc = mlir::FusedLoc::get(context, {allArgLocs});

// Mark load operations with unknownLoc or fusedLoc
for (auto& loadOp : printArgLoads) {
    loadOp.setLoc(rewriter.getUnknownLoc());
}

// Only the final print call gets the real location
printCall.setLoc(loc);
```

**Test:** Step through `print_multiple.silly` - should only stop once

**Estimated Effort:** 1 day

#### C) Fix Loop Body Line Jumping
**Problem:** Stepping through loop alternates between statements (TODO #41-72)

**Root Cause:** Unknown locations causing debugger confusion

**Solution:**
- Audit all `getUnknownLoc()` calls in lowering
- Replace with `FusedLoc` marked as "compiler-generated"
- Use proper source locations for user-written operations

**Test:** Step through `printdi.silly` loop - should step linearly

**Estimated Effort:** 2 days

**Total Effort for Debug Info:** 4 days

**Priority:** HIGH - Significantly impacts usability

---

### 3. ⭐ Runtime Bounds Checking (Optional Feature)

**Why:** Safety, better error messages than segfaults

**What:** Add `--bounds-check` compiler flag for array access validation

**Implementation:**
```cpp
// In driver.cpp:
struct CompilerOptions {
    bool boundsCheck = false;  // Add this flag
    // ... existing options
};

// In lowering, when generating array access:
if (options.boundsCheck) {
    // Before: %elem = load %array[%index]
    
    // After:
    %size = arith.constant <array_size> : i64
    %negative = arith.cmpi slt, %index, %c0 : i64
    %too_large = arith.cmpi sge, %index, %size : i64
    %out_of_bounds = arith.ori %negative, %too_large : i1
    
    scf.if %out_of_bounds {
        // Call runtime: __silly_abort_oob(file, line, index, size)
        func.call @__silly_abort_oob(%filename, %line, %index, %size)
    } else {
        %elem = load %array[%index]
    }
}
```

**Runtime Addition:**
```cpp
// src/runtime/Silly_runtime.cpp
extern "C" void __silly_abort_oob(const char* file, int line, 
                                   int64_t index, int64_t size) {
    fprintf(stderr, "%s:%d: error: array index %ld out of bounds [0, %ld)\n",
            file, line, index, size);
    abort();
}
```

**Usage:**
```bash
silly --bounds-check myprogram.silly
./myprogram
# If OOB access:
# myprogram.silly:42:5: error: array index 10 out of bounds [0, 5)
```

**Estimated Effort:** 2 days

**Priority:** MEDIUM-HIGH - Great safety feature, relatively easy

---

## 🔧 MEDIUM PRIORITY (Nice to Have)

### 4. Error Message Enhancements

**Current State:** Good, but can be improved

**A) Error Numbers (TODO #8)**
```
error_nested.silly:5:5: error E0042: Nested functions are not currently supported.
```

**Implementation:**
```cpp
enum class ErrorCode {
    E0001_UndeclaredVariable,
    E0002_TypeMismatch,
    E0042_NestedFunction,
    // ... etc
};

void emitUserError(mlir::Location loc, ErrorCode code, const std::string& msg) {
    llvm::errs() << formatLocation(loc) 
                 << ": error E" << std::setw(4) << std::setfill('0') << (int)code
                 << ": " << msg << "\n";
}
```

**Estimated Effort:** 1 day

**B) Error Limit (TODO #9)**
```cpp
// In emitUserError:
if (errorCount >= maxErrors) {
    llvm::errs() << "Too many errors (" << errorCount << "), stopping.\n";
    exit(1);
}
```

**Command Line:**
```bash
silly --max-errors=10 file.silly
silly --show-internal-errors file.silly
```

**Estimated Effort:** Half day

**C) Color Output (Optional)**
```cpp
namespace {
    const char* RED = isatty(fileno(stderr)) ? "\033[1;31m" : "";
    const char* CYAN = isatty(fileno(stderr)) ? "\033[0;36m" : "";
    const char* RESET = isatty(fileno(stderr)) ? "\033[0m" : "";
}

llvm::errs() << CYAN << filename << ":" << line << ":" << col << ": "
             << RED << "error: " << RESET << message << "\n";
```

**Estimated Effort:** Half day

**Total Effort:** 2 days

**Priority:** MEDIUM - Polish features

---

### 5. BREAK/CONTINUE Support

**Why:** Common programming pattern, users expect it

**Status:** Not supported (should document in README)

**Grammar Addition:**
```antlr4
breakStatement
  : BREAK_TOKEN ';'
  ;

continueStatement
  : CONTINUE_TOKEN ';'
  ;
```

**MLIR Generation:**
```cpp
// BREAK in FOR loop:
scf.for %i = %start to %end step %step {
    scf.if %condition {
        scf.yield  // break - exit loop
    }
    // ... loop body
}

THIS "break" code suggestion IS WRONG.  There's a different way, but it requires switching FOR to scf.while, which is more complicated -- see: tests/endtoend/for/forbreak.mlsilly

// CONTINUE:
scf.for %i = %start to %end step %step {
    scf.if %condition {
        scf.yield  // continue - next iteration
    }
    // ... loop body
    scf.yield  // normal loop continuation
}
```

**Challenges:**
- Need to track loop context in parser
- Generate correct `scf.yield` for each case
- Handle nested loops correctly

**Estimated Effort:** 2 days

**Priority:** MEDIUM - Useful feature, moderate complexity

---

### 6. WHILE Loop Support

**Why:** More natural for certain algorithms

**Grammar:**
```antlr4
whileStatement
  : WHILE_TOKEN '(' expression ')' '{' statement* '}' ';'
  ;
```

**MLIR:**
```mlir
scf.while () : () -> () {
  %cond = ...  // evaluate condition
  scf.condition(%cond)
} do {
^bb0():
  // loop body
  scf.yield
}
```

**Estimated Effort:** 2 days

**Priority:** MEDIUM - Nice to have, not essential

---

### 7. CMake Improvements

**Add Documentation Target (TODO: not in list but recommended):**
```cmake
# In CMakeLists.txt
find_package(Doxygen OPTIONAL_COMPONENTS dot)

if(DOXYGEN_FOUND)
    add_custom_target(docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_SOURCE_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Generating Doxygen documentation"
        VERBATIM
    )
endif()
```

**Usage:**
```bash
ninja docs
# or
make docs
```

**Estimated Effort:** 15 minutes

**Priority:** LOW - Convenience feature

---

## 🐛 KNOWN BUGS (From TODO)

### Bug #1: `error_intarray_bad_constaccess.silly` (TODO #79)
**Status:** Should fail compilation but doesn't

**Likely Cause:** No bounds checking in parser/semantic analysis

**Fix:** Implement in semantic analysis pass (#1 above)

**Priority:** HIGH - Part of semantic analysis

---

### Bug #2: Division by Zero Behavior (TODO #29)
**Status:** Different results on Intel vs ARM

**Example:**
```silly
INT32 x = 5 / 0;  // Intel: crashes, ARM: returns 0?
```

**Options:**
1. Document as undefined behavior
2. Add compile-time check (if divisor is constant 0)
3. Add runtime check with `--bounds-check`

**Priority:** LOW - Can document as UB for now

---

### Bug #3: Negative/Zero Step in FOR (TODO #30)
**Status:** Undefined behavior, no runtime check

**Test Case:** `negative_step_for.silly`, `zero_step_for.silly`

**Options:**
1. Document as UB (current)
2. Compile-time error if step is constant non-positive
3. Runtime check with `--bounds-check`

**Priority:** LOW - Document in README

---

### Bug #4: Parse Error for `error_invalid_unary` (TODO #31)
**Status:** Parse error doesn't trigger compile error

**Investigation Needed:** Why doesn't ANTLR error propagate?

**Priority:** MEDIUM - Error handling gap

---

## 📝 LANGUAGE IMPROVEMENTS (Low Priority)

### Forward Function Declarations (TODO #15)
**Why:** Allow calling functions before definition

**Grammar:**
```antlr4
functionDeclaration
  : FUNCTION_TOKEN ID_TOKEN '(' parameterList ')' (':' returnType)? ';'
  ;
```

**Challenges:**
- Need symbol table to track declarations vs definitions
- Verify all declarations have matching definitions
- Handle mutual recursion

**Estimated Effort:** 1 week

**Priority:** LOW - Nice to have

---

### Multi-Variable Declaration (TODO #16)
**Why:** Convenience

**Example:**
```silly
INT64 a = 1, b = 2, c = 3;  // Currently not supported
INT64 a{1}, b{2}, c{3};     // Also not supported
```

**Grammar Change:**
```antlr4
declareStatement
  : type declarator (',' declarator)* ';'
  ;

declarator
  : ID_TOKEN ('[' INTEGER ']')? ('=' expression | '{' initList '}')?
  ;
```

**Estimated Effort:** 2 days

**Priority:** LOW - Syntactic sugar

---

### Multi-Dimensional Arrays
**Example:**
```silly
INT32 matrix[3][4];  // Not supported
```

**Challenges:**
- Type system changes (array of arrays)
- Indexing syntax: `matrix[i][j]` vs `matrix[i, j]`
- Memory layout (row-major vs column-major)
- MLIR representation

**Estimated Effort:** 1-2 weeks

**Priority:** LOW - Complex feature

---

## 🔨 CODE QUALITY IMPROVEMENTS

### Refactor driver.cpp (TODO #12)
**Status:** "Spaghetti code mess" per TODO

**Current:** One large file with command-line parsing, compilation pipeline, linking

**Recommended Split:**
```
src/driver/
├── driver.cpp           # Main entry point
├── CommandLine.cpp      # Option parsing
├── CompilerPipeline.cpp # Parse → MLIR → Lower → Codegen
├── Linker.cpp           # Object file linking
└── Utils.cpp            # Shared utilities
```

**Estimated Effort:** 1 day

**Priority:** LOW - Works fine as-is, but would improve maintainability

---

### Split Large Files
**Current:**
- `lowering.cpp`: 2403 lines
- `parser.cpp`: 2273 lines

**Assessment:** Large but manageable with good organization

**Recommendation:** Split when >3000 lines or if adding major features

**Priority:** LOW - Not urgent

---

## 🧪 TESTING IMPROVEMENTS

### Automated Debug Info Tests (TODO #34-40)
**Current:** Manual testing with `dwarfdump`

**Goal:** Automate DWARF verification

**Example Test:**
```python
# tests/debug/test_dwarf.py
def test_for_induction_variable():
    compile('for_simplest.silly', debug=True)
    dwarf = parse_dwarfdump('out/for_simplest')
    
    var_i = dwarf.find_variable('i')
    assert var_i.type == 'int64_t'
    assert var_i.decl_line == 3
    assert var_i.scope.type == 'lexical_block'  # Not function scope!
```

**Estimated Effort:** 2-3 days to build infrastructure

**Priority:** MEDIUM - Prevents debug info regressions

---

### Test Categories to Add
- **Negative tests:** More error cases (semantic errors, type errors)
- **Performance tests:** Benchmark generated code quality
- **Fuzzing:** Random program generation to find crashes
- **Regression tests:** Preserve fixes for bugs #1-4

**Priority:** ONGOING - Continuous improvement

---

## 🚀 ADVANCED FEATURES (Future)

### Language Server Protocol (LSP)
**Features:**
- Hover for type info
- Go-to-definition
- Real-time diagnostics
- Code completion

**Estimated Effort:** 4-6 weeks

**Priority:** VERY LOW - Tooling project

---

### JIT Mode
**What:** Interpret silly programs without compilation

**Use:** REPL, scripting, faster iteration

**Estimated Effort:** 2-3 weeks

**Priority:** VERY LOW - Alternative execution mode

---

### Optimization Passes
**Examples:**
- Constant folding
- Dead code elimination  
- Loop optimizations
- Inlining

**Estimated Effort:** Ongoing

**Priority:** LOW - Performance isn't critical for toy language

---

### Multiple Backends
**Current:** LLVM only

**Potential:**
- SPIR-V (GPU execution)
- WebAssembly (browser)
- C transpilation (portability)

**Estimated Effort:** 2-4 weeks per backend

**Priority:** VERY LOW - LLVM is sufficient

---

## 📋 QUICK WINS (< 2 Hours Each)

These are small improvements you could knock out quickly:

1. ✅ **Add `ninja docs` target** (see #7 above) - 15 min
2. **Add `--version` flag** to silly compiler - 15 min
3. **Color error output** (optional, see #4C) - 30 min
4. **Create `CONTRIBUTING.md`** with development workflow - 1 hour
5. **Fix hardcoded `/build/` paths** (TODO #17-23) - 1 hour
6. **Add `.clang-tidy` config** for static analysis - 30 min
7. **Add `.editorconfig`** for consistent formatting - 15 min
8. **Create simple bash completion** for silly options - 1 hour

---

## 🗓️ RECOMMENDED 3-MONTH ROADMAP

### Month 1: Core Correctness
**Focus:** Semantic analysis and debug info

- **Week 1-2:** Semantic analysis pass (#1)
  - Implement `SemanticAnalyzer` class
  - Fix test cases that should fail
  - Add new semantic error tests
  
- **Week 3-4:** Debug info improvements (#2)
  - Fix loop variable scope
  - Consolidate PRINT locations  
  - Fix line-jumping in loops
  - Add automated DWARF tests

**Outcome:** Compiler catches more errors, debugging experience improved

---

### Month 2: Safety & Features
**Focus:** Runtime safety and language improvements

- **Week 1:** Runtime bounds checking (#3)
  - Add `--bounds-check` flag
  - Implement runtime checks
  - Add abort handler
  
- **Week 2:** Error message polish (#4)
  - Error numbers
  - Error limits
  - Optional color output
  
- **Week 3:** BREAK/CONTINUE (#5)
  - Grammar changes
  - Parser implementation
  - Test cases
  
- **Week 4:** WHILE loops (#6)
  - Grammar and parser
  - MLIR lowering
  - Test suite

**Outcome:** More robust, safer, feature-complete language

---

### Month 3: Polish & Documentation
**Focus:** Code quality and user experience

- **Week 1:** Fix known bugs
  - Bug #1: OOB access detection
  - Bug #4: Parse error propagation
  - Clean up TODOs
  
- **Week 2:** Code refactoring
  - Split driver.cpp
  - Clean up large functions
  - Improve comments
  
- **Week 3:** Documentation
  - Update README with new features
  - Write architecture guide
  - Create tutorial examples
  - Document undefined behaviors
  
- **Week 4:** Testing improvements
  - Automated DWARF tests
  - More negative test cases
  - Performance benchmarks

**Outcome:** Production-quality compiler with great documentation

---

## 📊 SUMMARY

### Completed Since Last Review ✅
- User-friendly error reporting (GCC/Clang style)
- Hierarchical test organization (178 tests)
- CMake install rules
- Doxygen documentation

### High Priority (Next 1-2 Months) ⭐⭐⭐
1. Semantic analysis pass (3-4 days)
2. Debug info quality (4 days)
3. Runtime bounds checking (2 days)

### Medium Priority (2-3 Months) ⭐⭐
4. Error message enhancements (2 days)
5. BREAK/CONTINUE (2 days)
6. WHILE loops (2 days)

### Low Priority (Future) ⭐
- Language features (multi-dim arrays, structs, etc.)
- Advanced tooling (LSP, JIT, optimizations)
- Code refactoring (split large files)

### Quick Wins (< 2 Hours) ✨
- Add `ninja docs` target
- Add `--version` flag
- Color error output
- Fix hardcoded build paths

---

## 🎯 IF YOU ONLY DO THREE THINGS

1. **Semantic Analysis Pass** (#1) - Catches errors earlier, professional-grade compiler
2. **Fix Debug Info** (#2) - Makes debugging pleasant instead of frustrating  
3. **Runtime Bounds Checking** (#3) - Safety feature, prevents mysterious segfaults

These three will have the biggest impact on compiler quality and user experience.

---

**Questions? Want detailed implementation plans for any item?** Feel free to ask!

<!-- vim: set tw=80 ts=2 sw=2 et: -->
