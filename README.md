# Overview

## Motivation

The goal of this project was to gain concrete, hands-on experience with the MLIR ecosystem.
It uses an ANTLR4 grammar, an MLIR builder, and MLIR lowering to LLVM IR, incorporating a custom dialect (silly) along with several existing MLIR dialects (scf, arith, memref, etc.).

### Why MLIR?

My interest in MLIR was driven by work with proprietary AST walkers in both Clang and proprietary commercial compilers.
In some cases, AST walking infrastructure can be used to extend the compiler itself by adding user-defined semantics tailored to a specific customer base,
and is immensely powerful and valuable.
While Clang's AST walk and rewrite API doesn't allow for user-defined language extension plugins,
having a structured representation of source code that can be queried programmatically is amazing and very useful.

What does that have to do with MLIR? I had seen MLIR in action in a prototype project at a previous job, and that prototype suggested that it was a natural way to avoid hand-coding an AST.
An MLIR representation of a program could:
* drive a semantic analysis pass,
* perform code generation,
* enable in-language transformation passes, and
* allow for cross-dialect (cross-language) transformations.

In short, I was intrigued enough to explore MLIR personally, even without a work-related justification, and this project was born.

### What is this project?

Initially, I used MLIR to build a simple symbolic calculator that supported double-like variable declarations, assignments, unary and binary arithmetic operations, and output display.
This used a dialect initially called toy, but that dialect was renamed to silly to avoid confusion with the MLIR tutorial toy dialect.

As noted earlier, the primary goal of the project was not calculation itself, but to gain concrete, hands-on experience with the MLIR ecosystem.

That initial implementation has evolved into a silly language and its compiler.
There's no good reason to use this language, nor the compiler, but it was fun to build.

## Language Overview

The language and compiler now support the following features:

### Variable Declarations

**Floating-point types** (`FLOAT32`, `FLOAT64`):
```silly
FLOAT32 pi = 3.14;
FLOAT64 e{2.7182};
FLOAT64 a[3];  // uninitialized array
```

**Integer types** (`INT8`, `INT16`, `INT32`, `INT64`):
```silly
INT32 answer = 42;
INT64 bignum{12345};
INT16 values[3]{10, 20, 30};  // initialized array
```

**Boolean type** (`BOOL`):
```silly
BOOL flag = TRUE;
BOOL bits[2]{TRUE, FALSE};
```

**Legacy `DECLARE` operation** (implicit double type, may be removed):
```silly
DECLARE x;  // equivalent to: FLOAT64 x;
```

### Input/Output Operations

**`PRINT`** — outputs to stdout:
```silly
PRINT "Hello, world!";
PRINT "x = ", x;
PRINT arr[0], ", ", arr[1];
PRINT 3.14 CONTINUE;  // suppress trailing newline
```

**`ERROR`** — outputs to stderr:
```silly
ERROR "Warning: value out of range";
ERROR "Got: ", invalid_value;
```

**`GET`** — reads from stdin:
```silly
GET x;          // read into scalar
GET arr[i];     // read into array element
```

### Comments

Single-line comments:
```silly
// This is a comment
INT32 x = 42;  // inline comment
```

### Control Flow

**`IF`/`ELIF`/`ELSE`** — conditional execution:
```silly
IF (x < 0) {
    PRINT "negative";
} ELIF (x EQ 0) {
    PRINT "zero";
} ELSE {
    PRINT "positive";
};
```

**`FOR`** — range-based loop with optional step:
```silly
// Basic loop from 1 to 10 (inclusive)
FOR (INT32 i : (1, 10)) {
    PRINT i;
};

// With step size
FOR (INT32 j : (0, 100, 5)) {
    PRINT j;  // prints 0, 5, 10, ..., 100
};
```

The induction variable is privately scoped to the loop.

### Functions

**Function definition** with typed parameters and optional return type:
```silly
// Function with return value
FUNCTION add(INT32 a, INT32 b) : INT32 {
    RETURN a + b;
};

// Void function
FUNCTION greet(INT32 count) {
    FOR (INT32 i : (1, count)) {
        PRINT "Hello #", i;
    };
    RETURN;
};

// Recursion supported
FUNCTION factorial(INT32 n) : INT32 {
    INT32 result = 1;
    IF (n > 1) {
        result = n * CALL factorial(n - 1);
    };
    RETURN result;
};
```

**Function calls** using the `CALL` keyword:
```silly
INT32 sum = CALL add(10, 20);
CALL greet(3);

// CALL can be used in expressions
INT32 x = 1 + CALL add(2, 3) * 4;
```

### Expressions

**Generalized expressions** with full operator precedence, parentheses, and unary chaining:
```silly
INT32 result = ((a + b) * c - d) / e;
FLOAT64 x = -CALL abs(y) + 3.14;
BOOL valid = (x > 0) AND (x < 100);
```

**Supported operators** (by category):
- Arithmetic: `+`, `-`, `*`, `/`
- Comparison: `<`, `>`, `<=`, `>=`, `EQ`, `NE`
- Logical/Bitwise: `AND`, `OR`, `XOR`, `NOT`
- Unary: `+`, `-`, `NOT`

All operators support proper precedence and associativity rules.

### Assignment

**Assignment operator** (`=`) for scalars and array elements:
```silly
x = 42;
arr[0] = x * 2;
y = a + b * c;  // expression assignment
```

### Literals

**Numeric literals** (integer and floating-point):
```silly
INT32 i = 42;
FLOAT64 pi = 3.14159;
FLOAT32 sci = 2.5E-3;
```

**Boolean literals**:
```silly
BOOL yes = TRUE;
BOOL no = FALSE;
```

**String literals**:
```silly
INT8 msg[6] = "Hello";  // STRING is alias for INT8[]
PRINT "Embedded string";
```

### Arrays

**Array support** including declaration, initialization, access, and element-wise operations:
```silly
INT32 arr[5];           // uninitialized
INT32 init[3]{1,2,3};   // initialized
arr[i] = 42;            // assignment
PRINT arr[2];           // element access
EXIT arr[0];            // use in expressions
```

### Program Termination

**`EXIT`** — explicit program exit with optional status code:
```silly
EXIT;           // exit with status 0
EXIT 0;         // explicit success
EXIT 1;         // exit with error code
EXIT status;    // exit with variable value
```

**`ABORT`** — emergency termination with error message:
```silly
IF (error_condition) {
    ERROR "Fatal error detected";
    ABORT;  // prints location and aborts
};
```

### Debugging Support

**DWARF instrumentation** sufficient for:
- Line stepping in gdb/lldb
- Breakpoints
- Variable inspection
- Call stack unwinding

Variable modification is likely supported but untested.

---

There is lots of room to add further language elements to make the compiler and language more interesting.
Some ideas for improvements (as well as bug fixes) can be found in TODO.md

## Language Quirks and Bugs

* Like scripted languages, there is an implicit `main` in this silly language.
No user-defined `main` function is allowed.
* Functions can be defined anywhere, but must be defined before use (no forward declarations).
* The `EXIT` statement, if specified, currently must be at the end of the program.
`EXIT` without a numeric value is equivalent to `EXIT 0`, as is a program with no explicit `EXIT`.
* The `RETURN` statement must be at the end of a function.
It is currently mandatory.
* See TODO.md for a long list of nice-to-have features that I haven't gotten around to yet, and may never.
* `GET` into a `BOOL` value will abort if the input value is not 0 or 1.
This is inconsistent with assignment to a `BOOL` variable, which will truncate without raising a runtime error.
* Variables declared in `FOR` or `IF` conditions persist past the block that declared them, as if they were declared in the enclosing scope of the function.
For example, these are equivalent:
```silly
FUNCTION foo() {
    INT32 x = 1;
    PRINT x;
    RETURN;
};

FUNCTION foo() {
    IF (TRUE) {
        INT32 x = 1;
    };
    PRINT x;  // x is still accessible here
    RETURN;
};
```
* The storage requirement of `BOOL` is currently one byte per element, even for arrays.
Array `BOOL` values may use a packed bitmask representation in the future.

## Interesting Files

* `src/grammar/Silly.g4` — The ANTLR4 grammar
* `src/dialect/silly.td` — MLIR dialect definition: the compiler's internal view of all grammar elements
* `src/driver/driver.cpp` — Compiler driver: handles command-line options, opens output files, and orchestrates all lower-level actions (parse tree walk + MLIR builder, lowering to LLVM IR, assembly printing, and linker invocation)
* `src/driver/parser.cpp` — ANTLR4 parse tree walker and MLIR builder
* `src/driver/lowering.cpp` — LLVM IR lowering classes
* `tests/endtoend/*.silly` and `bin/testit` — Sample programs and a rudimentary regression test suite
* `bin/build`, — Build script: `build` runs CMake and Ninja (sets compiler overrides if required)
* `bin/silly-opt` — Wrapper for mlir-opt that loads the silly dialect shared object
* `tests/dialect/*.mlir` — Silly dialect-level error checking for verify functions

## Command Line Options

Once built, the compiler driver can be run with `build/bin/silly` with the following options:

### User Options

* `-g` — Show MLIR location info in dumps and lowered LLVM IR
* `-O[0123]` — Optimization level (standard)
* `-c` — Compile only (don't link)
* `--init-fill nnn` — Set fill character for stack variables (numeric value ≤ 255). Default is zero-initialized.
* `--output-directory` — Specify output directory for generated files

### Debug/Hacking Options

* `--emit-llvm` — Emit LLVM IR files
* `--emit-mlir` — Emit MLIR files
* `--debug` — Enable MLIR debug output (built-in option)
* `-debug-only=silly-driver` — Enable driver-specific debug output
* `-debug-only=silly-lowering` — Enable lowering-specific debug output
* `--debug-mlir` — Enable MLIR-specific debugging
* `--stdout` — Send MLIR and LLVM IR output to stdout instead of files
* `--no-emit-object` — Skip object file generation

### Examples

```bash
cd tests/endtoend
rm -rf out
mkdir out
../../build/bin/silly --output-directory out f.silly -g --emit-llvm --emit-mlir --debug
../../build/bin/silly --output-directory out f.silly -O2
```

## Running silly-opt

Example:

```bash
cd tests/endtoend
testit -j loadstore.silly
silly-opt --pretty --source out/loadstore.mlir
```

By default, silly-opt output goes to stdout.
Run `silly-opt --help` for all available options.

## Building

### ANTLR4 Setup (Ubuntu)

```bash
sudo apt-get install libantlr4-runtime-dev antlr4
```

This assumes that the ANTLR4 runtime, after installation, is version 4.10.

On WSL2/Ubuntu, the installed runtime version may not match the generator version.
Workaround:

```bash
wget https://www.antlr.org/download/antlr-4.10-complete.jar
```

### Installation Dependencies (Fedora)

```bash
sudo dnf -y install antlr4-runtime antlr4 antlr4-cpp-runtime antlr4-cpp-runtime-devel \
    cmake clang-tools-extra g++ ninja cscope clang++ ccache libdwarf-tools
```

### Building MLIR

On both Ubuntu and Fedora, I needed a custom build of LLVM/MLIR, as I didn't find a package that included the MLIR TableGen files.
As it turned out, a custom LLVM/MLIR build was also required to specifically enable RTTI, as ANTLR4 uses `dynamic_cast<>`.
The `-fno-rtti` flag required by default to avoid typeinfo symbol link errors explicitly breaks the ANTLR4 header files.
This could be avoided by separating the ANTLR4 listener class from the MLIR builder, but the MLIR builder effectively
provides an AST, which I don't need to build separately if I construct it directly from the listener.

Question: Does the llvm-project have a generic lexer/parser, or do Clang/Flang/etc. each roll their own?
Having used ANTLR4 for previous prototyping (also generating C++ listeners), it made sense to use what I knew.

See `bin/buildllvm` for how I built and deployed the LLVM+MLIR installation used for this project.

The currently required version of LLVM/MLIR is:

    21.1.0-rc3

Any 21.1.* version after that will probably work as well.

### Building the Project

```bash
. ./bin/env
build
```

The build script currently assumes that I'm the one building it, and is likely not sufficiently general for other people to use. It will surely break as I upgrade the systems I build on.

Linux-only is assumed.

Depending on what I currently have booted, this project has been built on only a few configurations:

* Fedora 42/X64 (on a dual-boot Windows 11/Linux laptop)
* WSL Ubuntu 24/X64 (same laptop)
* Armbian (Ubuntu), running on a Raspberry Pi (this is why there's an ARM case in buildllvm and CMakeLists.txt)

### Testing

Testing is ctest-based. Examples:

```bash
cd build
ctest -j 3                      # Run the full test suite
ctest -R EndToEnd --verbose     # Run all the tests/endtoend/ tests.
ctest -R silly-dialecttests     # Run the low-level dialect verify tests (tests/dialect/)
```

---

# Silly Language — Operations Reference

The following describes the **operations and statements** supported by the *Silly* language, as defined by the `Silly.g4` ANTLR4 grammar.
It is intended as a language-level reference rather than a grammar walkthrough.

## Program Structure

A Silly program consists of zero or more **statements** and **comments**, optionally followed by an explicit `EXIT` statement.
Each statement is terminated by a semicolon (`;`).

Blocks use `{ ... }`, and expressions use parentheses `( ... )`.

---

## Expressions

The expression grammar supports **generalized expressions** with proper precedence, associativity, and parentheses.

### Operator Precedence (highest to lowest)

| Precedence | Operators                  | Associativity | Notes                               |
|------------|----------------------------|---------------|-------------------------------------|
| 1 (highest)| `NOT`, unary `+`, unary `-`| right         | Unary operators chain right-to-left |
| 2          | `*`, `/`                   | left          | Multiplicative                      |
| 3          | `+`, `-`                   | left          | Additive                            |
| 4          | `<`, `>`, `<=`, `>=`       | left          | Relational (non-associative in practice) |
| 5          | `EQ`, `NE`                 | left          | Equality                            |
| 6          | `AND`                      | left          | Bitwise/logical AND                 |
| 7          | `XOR`                      | left          | Bitwise/logical XOR                 |
| 8 (lowest) | `OR`                       | left          | Bitwise/logical OR                  |

### Parentheses

Parentheses override default precedence:

```text
x = (a + b) * c;
```

### Examples

```text
// Arithmetic
result = 2 + 3 * 4;        // = 14
result = (2 + 3) * 4;      // = 20

// Comparisons
valid = x > 0 AND x < 100;

// Unary chaining
y = - - x;                 // double negation
flag = NOT NOT condition;

// Complex expression
z = (a + b) * (c - d) / e;
```

---

## Variable Declarations

Variables are declared with an explicit type and optional initializer or optional assignment.
Array variables may only using the optional initializer syntax.

### Scalar Declarations

```text
TYPE name;
TYPE name = initializer;
TYPE name{initializer};
TYPE name{};
```

An un-initialized variable gets a default value from the `--init-fill` command line option.
That value defaults to zero.

Specification of an empty initializer expression results in zero initialization (not the `--init-fill` value.)

### Array Declarations

```text
TYPE name[size];
TYPE name[size]{val1, val2, ...};
```

- **Arrays** must have a compile-time constant size
- **Initializer lists** can be shorter than the array (rest are explicitly zero-initialized, and do not use the `--init-fill` value.)
- **Initializer expressions** must be constant expressions.
Initializers that references variables have undefined behaviour.
- **Excess initializers** are an error

### Supported Types

| Type      | Description          | Width  |
|-----------|----------------------|--------|
| `BOOL`    | Boolean              | 8 bits |
| `INT8`    | Signed integer       | 8 bits |
| `INT16`   | Signed integer       | 16 bits|
| `INT32`   | Signed integer       | 32 bits|
| `INT64`   | Signed integer       | 64 bits|
| `FLOAT32` | Floating-point       | 32 bits|
| `FLOAT64` | Floating-point       | 64 bits|

### Examples

```text
INT32 x;
FLOAT64 pi = 3.14159;
BOOL flags[10];
INT32 values[3]{1, 2, 3};
```

---

## Assignment

Assigns a value to a variable or array element.

```text
variable = expression;
array[index] = expression;
```

Right-hand sides may be:

- Literals
- Variables or array elements
- Unary expressions
- Binary expressions
- Function calls
- Allowed combinations of the above, using parenthesis where desired.

---

## Literals

### Numeric Literals

```text
42
-7
3.14
2.0E-3
```

### Boolean Literals

```text
TRUE
FALSE
```

(Boolean literals may also be represented numerically as 0 or 1.)

### String Literals

```text
"hello world"
```

---

## Unary Operations

Unary operators apply to scalar values or array elements.

| Operator | Meaning           |
|----------|-------------------|
| `+`      | Unary plus        |
| `-`      | Unary negation    |
| `NOT`    | Boolean negation  |

```text
x = -y;
flag = NOT flag;
```

---

## Binary Arithmetic Operations

Binary arithmetic operators work on numeric operands.

| Operator | Meaning        |
|----------|----------------|
| `+`      | Addition       |
| `-`      | Subtraction    |
| `*`      | Multiplication |
| `/`      | Division       |

```text
x = a + b;
y = x * 3;
```

---

## Comparison (Predicate) Operations

Comparison operators produce boolean values.

| Operator | Meaning                |
|----------|------------------------|
| `<`      | Less than              |
| `>`      | Greater than           |
| `<=`     | Less than or equal     |
| `>=`     | Greater than or equal  |
| `EQ`     | Equal                  |
| `NE`     | Not equal              |

```text
IF (x < 10) { PRINT x; };
```

These operators work across any combination of floating-point and integer types (including `BOOL`).

---

## Boolean Operations

Boolean operators may be logical or bitwise depending on operand types.

| Operator | Meaning      |
|----------|--------------|
| `AND`    | Boolean AND  |
| `OR`     | Boolean OR   |
| `XOR`    | Boolean XOR  |

```text
flag = a AND b;
```

Integer bitwise operators (`OR`, `AND`, `XOR`) are applicable only to integer types (including `BOOL`).

---

## Control Flow

### IF / ELIF / ELSE

Conditional execution using boolean expressions.

```text
IF (x < 0) {
    PRINT "negative";
} ELIF (x EQ 0) {
    ERROR "zero";
    ABORT;
} ELSE {
    PRINT "positive";
};
```

---

### FOR Loop

Range-based iteration with optional step size (must be positive).

**Note:** There is currently no checking that the step size is positive or non-zero.
Use of a negative or zero step has undefined behavior.

```text
FOR (INT32 i : (1, 10)) {
    PRINT i;
};

// With expressions for range and step
INT32 a = 0;
INT32 b = -20;
INT32 c = 2;
INT32 z = 0;
FOR (INT32 i : (+a, -b, c + z)) {
    PRINT i;
};
```

Semantically equivalent to:

```c
{
    int i = start;
    while (i <= end) {
        ...
        i += step;
    }
}
```

**Constraints:**
- The induction variable name must not be used by any variable in the function
- It cannot shadow any induction variable of the same name in an outer FOR loop

---

## Functions

### Definition

Defines a function with typed parameters and optional return type.

```text
FUNCTION add(INT32 a, INT32 b) : INT32 {
    RETURN a + b;
};

FUNCTION void(INT32 a, INT32 b) {
    PRINT a + b;
    RETURN;
};
```

**Notes:**
- Parameters and return types must be scalar types
- A `RETURN` statement is required and must be the last statement
- Recursion is supported

---

### Function Calls

Functions are invoked using the `CALL` keyword.

```text
x = CALL add(2, 3);
y = CALL sum(a[1], a[2]);
```

CALL expressions can be part of larger expressions:

```text
x = 1 + v * CALL foo();
x = - CALL foo();
```

---

## Input and Output

### PRINT

Outputs one or more expressions (variables, literals, array elements, expressions) with a trailing newline (unless `CONTINUE` is specified).

```text
PRINT x, " is ", y;
PRINT 3.14 CONTINUE;
PRINT "Hello: ", v;
PRINT arr[3];
PRINT "hi", s, 40 + 2, ", ", -x, ", ", f[0], ", ", CALL foo();
```

### ERROR

The `ERROR` statement is equivalent to `PRINT`, but outputs to stderr instead of stdout.

```text
ERROR "Unexpected value: " CONTINUE;
ERROR v;
```

### GET

Reads input into a scalar or array element.

```text
GET x;
GET arr[2];
```

---

## Program Termination

### EXIT

Explicitly terminates program execution, optionally returning a value.

```text
EXIT;
EXIT 0;
EXIT 39 + 3;
EXIT status;
EXIT arr[0];
```

### ABORT

Prints a message like `<file>:<line>:ERROR: aborting` to stderr, then aborts.

```text
ABORT;
```

---

## Comments

Single-line comments begin with `//` and extend to the end of the line.

```text
// This is a comment
```

---

## Summary of Core Operations

- Variable declaration (implicit-float and explicit types), scalars or arrays
- Scalar and array element assignment
- String variables and literals
- General expressions with standard precedence, associativity, and parentheses
- Boolean logic and comparisons
- Conditional execution (`IF` / `ELIF` / `ELSE`)
- Range-based `FOR` loops
- Functions and calls (with recursion support)
- Input (`GET`) and output (`PRINT`, `ERROR`)
- Explicit program termination (`EXIT`, `ABORT`)

---
