# Overview

## Motivation

The goal of this project was to gain concrete, hands-on experience with the MLIR ecosystem.
It uses an ANTLR4 grammar, an MLIR builder, and MLIR lowering to LLVM IR, incorporating a custom dialect (silly) along with several existing MLIR dialects (scf, arith, memref, etc.).

I had seen MLIR in action in a prototype project at work but had not worked with it directly.
It appeared to provide a structured mechanism that avoids the need to hand-craft an AST, while offering built-in semantic checking and a robust representation of source code that can serve as the basis for transformations.
The MLIR white paper discusses the pitfalls of premature lowering, which I recall focused primarily on lowering to object code.
However, a good high-level program representation can also enable high-level transformations, including those between languages, as well as code query tasks.
I have used the Clang AST API for querying code, generating code, and performing transformations, including in very large codebases where automating large structural changes is challenging.

I have also worked with proprietary AST walkers in commercial compilers, where such infrastructure can extend the compiler itself by adding user-defined semantics tailored to a specific customer base.
These tools are immensely powerful and valuable.
Having a structured representation of source code with an associated language therefore has strong appeal.
The potential of this approach is clear and exciting enough to explore personally, even without a work-related justification.

## What is this project?

Initially, I used MLIR to build a simple symbolic calculator that supported double-like variable declarations, assignments, unary and binary arithmetic operations, and output display.  This used a dialect initially called toy, but renamed to silly to avoid confusion with the mlir tutorial toy dialect.

As noted earlier, the primary goal of the project was not calculation itself, but to gain concrete, hands-on experience with the MLIR ecosystem.

That initial implementation has evolved into a silly language and its compiler.  There's no good reason to use this language, nor the compiler, but it was fun to build.  The language and compiler now support the following features:

* A `DECLARE` operation (implicit double type).
* Fixed-size integer declarations (`INT8`, `INT16`, `INT32`, `INT64`).
* Floating-point declarations (`FLOAT32`, `FLOAT64`).
* Boolean declaration (`BOOL`).
* A `PRINT` operation for printing to standard output.
* An `ERROR` operation for printing to standard error.
* A `GET` operation for reading from standard input.
* Single-line comments.
* An `EXIT` operation.
* An `ABORT` operation for program termination.
* Boolean, integer, and floating-point constants, along with expression evaluation.
* An ASSIGNMENT operator (`=`) with unary (`+`, `-`) and binary operators (`+`, `-`, `*`, `/`).
* DWARF instrumentation sufficient for line stepping, breakpoints, continue, and variable inspection (variable modification is likely supported but untested).
* Comparison operators (`<`, `<=`, `==`, `!=`) yielding `BOOL` values. These work across any combinations of floating-point and integer types (including `BOOL`).
* Integer bitwise operators (`OR`, `AND`, `XOR`), applicable only to integer types (including `BOOL`).
* A `NOT` operator yielding `BOOL`.
* Array support, including declaration, assignment, printing, returning, exiting, and element access.
* A `STRING` type as an alias for `INT8` arrays, with string literal assignment and `PRINT` implemented.
* User-defined functions. Calls use the form `CALL function_name(p1, p2)` or with assignment `x = CALL function_name(p1, p2)`. Declarations use: `FUNCTION foo(type name, type name, ...) : RETURN-type { ... ; RETURN v; };` (where : `RETURN-type` is optional).
* `IF`/`ELSE` statement support. Logical operators (`AND`, `OR`, `XOR`) are not supported in predicates (only comparisons like `<`, `>`, `<=`, `>=`, etc.). Complex predicates (e.g., `(a < b) AND (c < d)`) are not supported. Nested `IF`s are untested and may or may not work.
* A `FOR` loop (supporting start, end, and step-size params, and requiring a previously defined variable for the loop induction variable.)

There is lots of room to add add further language elements to make the compiler and language more interesting.  Some ideas for improvements (as well as bug fixes) can be found in TODO.md

## Language Quirks and Bugs.

* Like scripted languages, there is an implicit `main` in this silly language.
* Functions can be defined anywhere, but must be defined before use.
* Computations occur in assignment operations, and any types are first promoted to the type of the variable.
This means that `x = 1.99 + 2.99` has the value `3`, if `x` is an integer variable, but `4.98` if x is a `FLOAT32` or `FLOAT64`.
* The `EXIT` statement currently has to be at the end of the program.
`EXIT` without a numeric value is equivalent to `EXIT 0`, as is a program with no explicit `EXIT`.
* The RETURN statement has to be at the end of a function.  It is currently mandatory.
* See TODO.md for a long list of nice to have features that I haven't gotten around to yet, and may never.
* `GET` into a `BOOL` value will abort if the value isn't one of 0, or 1.  This is inconsistent with assignment to a BOOL variable, which will truncate and not raise a runtime error.
* Variables declared in FOR or IF conditions persist past the block that declared them, as if they were declared in the enclosing scope of the function.  For example, these are equivalent
```
FUNCTION foo()
{
    INT32 x = 1;
    PRINT x;
    RETURN;
};

FUNCTION foo()
{
    IF (TRUE)
    {
        INT32 x = 1;
    };
    PRINT x;
    RETURN;
};
```

## On the use of AI in this project.

AI tools (Grok and ChatGPT) were used to generate some of the initial framework for this project (April 2025 timeframe.)
At that point in time, considerable effort was required to keep both Grok and ChatGPT from hallucinating MLIR or LLVM APIs that don't exist,
but both of those models were invaluable for getting things started.

As an example of the pain of working with AI tools, here's a trivial example: I asked Grok to add comments to my grammar and fix the indenting, but it took 20 minutes to coerce it to use the grammar that I asked it to use (as it claims the ability to read internet content), but it kept making stuff up and injecting changes to the semantics and making changing grammar element name changes that would have broken my listener class.

## Interesting files

* `Silly.g4`           -- The Antlr4 grammar.
* `src/driver.cpp`   -- This is the compiler driver, handles command line options, opens output files, and orchestrates all the lower level actions (parse tree walk + MLIR builder, lowering to LLVM-IR, assembly printer, and calls the linker.)
* `src/silly.td` -- This is the MLIR dialect that defines the compiler eye view of all the grammar elements.
* `src/parser.cpp`   -- This is the Antlr4 parse tree walker and the MLIR builder.
* `src/lowering.cpp` -- LLVM-IR lowering classes.
* `prototypes/simplest.cpp`  -- A MWE with working DWARF instrumentation.  Just emits LLVM-IR and has no assembly printing pass like the silly compiler.
* `samples/*.silly` and `bin/testit` -- sample programs and a rudimentary regression test suite based on them.
* `bin/build`, `bin/rebuild` -- build scripts (first runs cmake and ninja and sets compiler override if required), second just ninja with some teeing and grepping.

## Command line options

Once built, the compiler driver can be run with `build/silly` with options including the following user options:

* `-g` (show MLIR location info in the dump, and lowered LLVM-IR.)
* `-O[0123]` -- the usual.
* `-c` (compile only, and don't link.)
* `--init-fill nnn` set the fill character for stack variables (should be a numeric value <= 255).  Default is zero initialized.
* `--output-directory`

and the following hacking/debug options:
* `--emit-llvm`
* `--emit-mlir`
* `--debug` (built in MLIR option.)
* `-debug-only=silly-driver`
* `-debug-only=silly-lowering`
* `--debug-mlir`
* `--stdout`.  MLIR and LLVM-IR output to stdout instead of to files.
* `--no-emit-object`

Examples

```
cd samples
rm -rf out
mkdir out
../build/silly --output-directory out f.silly -g --emit-llvm --emit-mlir --debug
../build/silly --output-directory out f.silly -O 2
```

## Building

### anltlr4 setup (ubuntu)

```
sudo apt-get install libantlr4-runtime-dev
sudo apt-get install antlr4
```

This assumes that the antlr4 runtime, after installation, is 4.10 -- if not, change appropriately (update bin/runantlr)

On WSL2/ubuntu, this will result in the installed runtime version not matching the generator.  Workaround:

```
wget https://www.antlr.org/download/antlr-4.10-complete.jar
```

### Installation dependencies (Fedora)

```
sudo dnf -y install antlr4-runtime antlr4 antlr4-cpp-runtime antlr4-cpp-runtime-devel cmake clang-tools-extra g++ ninja cscope clang++ ccache libdwarf-tools
```

### Building MLIR

On both ubuntu and fedora, I needed a custom build of llvm/mlir, as I didn't find a package that had the MLIR tblgen files.
As it turned out, a custom llvm/mlir build was also required to specifically enable rtti, as altlr4 uses `dynamic_cast<>`.
The -fno-rtti that is required by default to avoid typeinfo symbol link errors, explicitly breaks the antlr4 header files.
That could be avoided if I separated out the antlr4 listener class from the MLIR builder, but the MLIR builder effectively
provides an AST, which I don't need to build myself if I do the builder directly from the listener.

Question: Does the llvm-project has a generic lexer/parser, or does clang/flang/anything-else roll their own?
Having used antlr4 for previous prototyping, also generating a C++ listener, it made sense to me to use what I knew.

See `bin/buildllvm` for how I built and deployed the llvm+mlir installation used for this project.

The current required version of LLVM/MLIR is:

    21.1.0-rc3

Any 21.1.* version after that will probably work too.

### Building the project.

```
. ./bin/env
build
```

The build script current assumes that I'm the one building it, and is likely not sufficiently general for other people to use, and will surely break as I upgrade the systems I attempt to build it on.

Linux-only is assumed.

Depending on what I currently have booted, this project has been built on only a few configurations:

* Fedora 42/X64 (on a dual boot windows-11/Linux laptop)
* WSL ubuntu 24/X64 (same laptop.)
* Ambian (ubuntu), running on an raspberry PI (this is why there is an ARM case in buildllvm and CMakeLists.txt)

# Silly Language — Operations Reference

This following describes the **operations and statements** supported by the *Silly* language, as defined by the `Silly.g4` ANTLR4 grammar. It is intended as a language-level reference rather than a grammar walkthrough.

---

## Program Structure

A Silly program consists of zero or more **statements** and **comments**, optionally followed by an explicit `EXIT` statement. Each statement is terminated by a semicolon (`;`).

Blocks use `{ ... }`, and expressions use parentheses `( ... )`.

---

## Declarations

### Implicit floating point Declaration

Declares a scalar or array variable with an implicit `FLOAT64` type, with optional initialization.

```text
DCL x;
DECLARE y[10];
DCL y = 42.0;
```

### Typed Declarations

Scalar and array declarations with explicit types and optional initialization:

#### Integer Types

```text
INT8   a;
INT16  b;
INT32  c[4];
INT64  d = 42;
```

#### Floating-Point Types

```text
FLOAT32 f;
FLOAT64 g[8];
FLOAT32 pi = 3.14;
```

#### Boolean Type

```text
BOOL flag;
BOOL b = FALSE;
```

#### String Type

Strings must be arrays (fixed-length).

```text
STRING name[32];
STRING hello[32] = "world";
```

## Array Element Access

Array elements are accessed using square brackets:

```
a[0]
values[i]
values[i + j]
```

Array elements may be used wherever a scalar value is expected:

- Assignment rvalues
- Expressions
- Function call parameters
- PRINT, ERROR, GET, RETURN, and EXIT statements

They behave as **rvalues** in expressions and as **lvalues** on the left-hand side of assignments.

```
x = a[3];
y = b[i] + 2;
a[i] = 7;
PRINT arr[5];
ERROR arr[5];
GET arr[0];
EXIT arr[1];
```

Index expressions may be any integer valued assignment-rvalues.

---

## Assignment

Assigns a value or expression to a scalar or array element.

```text
x = 42;
a[3] = b;
y = -x;
```

Right-hand sides may be:

- literals
- variables or array elements
- unary expressions
- binary expressions
- function calls

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

(Boolean literals may also be represented numerically.)

### String Literals

```text
"hello world"
```

---

## Unary Operations

Unary operators apply to scalar values or array elements.

| Operator | Meaning |
|--------|---------|
| `+` | Unary plus |
| `-` | Unary negation |
| `NOT` | Boolean negation |

```text
x = -y;
flag = NOT flag;
```

---

## Binary Arithmetic Operations

Binary arithmetic operators work on numeric operands.

| Operator | Meaning |
|--------|---------|
| `+` | Addition |
| `-` | Subtraction |
| `*` | Multiplication |
| `/` | Division |

```text
x = a + b;
y = x * 3;
```

---

## Comparison (Predicate) Operations

Comparison operators produce boolean values.

| Operator | Meaning |
|--------|---------|
| `<`  | Less than |
| `>`  | Greater than |
| `<=` | Less than or equal |
| `>=` | Greater than or equal |
| `EQ` | Equal |
| `NE` | Not equal |

```text
IF (x < 10) { PRINT x; };
```

---

## Boolean Operations

Boolean operators may be logical or bitwise depending on operand types.

| Operator | Meaning |
|--------|---------|
| `AND` | Boolean AND |
| `OR`  | Boolean OR |
| `XOR` | Boolean XOR |

```text
flag = a AND b;
```

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

Range-based iteration with optional step size.

```text
INT32 i;
FOR ( i : (1, 10) ) {
  PRINT i;
};

INT32 a;
INT32 b;
INT32 c;
INT32 z;
a = 0;
b = -20;
c = 2;
z = 0;
FOR ( i : (+a, -b, c + z) ) {
  PRINT i;
};
```

Semantically equivalent to:

```text
i = start;
while (i <= end) {
  ...
  i += step;
}
```

---

## Functions

### Definition

Defines a function with typed parameters and optional return type.

```text
FUNCTION add(INT32 a, INT32 b) : INT32 {
  RETURN a + b;
};
```

Notes:

- Parameters and return types must be scalar types
- A `RETURN` statement is required

---

### Function Calls

Functions are invoked using the `CALL` keyword.

```text
x = CALL add(2, 3);
y = CALL sum(a[1], a[2]);
```

CALL expressions can also be part of unary or binary expressions, for example
```
x = v * CALL foo();
x = - CALL foo();
```

---

## Input and Output

### PRINT

Outputs one or more expressions (variables, literals, array elements, expressions) and a trailing newline after the last expression (unless CONTINUE is specified.)

```text
PRINT x, " is ", y;
PRINT 3.14 CONTINUE;
PRINT "Hello: ", v;
PRINT arr[3];
PRINT "hi", s, 40 + 2, ", ", -x, ", ", f[0], ", ", CALL foo();
```

### ERROR

The ERROR statement is equivalent to PRINT, but prints to stderr instead of stdout.

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

Prints a message like `<file>:<line>:ERROR ERROR: aborting` to stderr, and then aborts.

---

## Comments

Single-line comments begin with `//` and extend to the end of the line.

```text
// This is a comment
```

---

## Summary of Core Operations

- Variable declaration (implicit-float and explicit-types), scalar or arrays.
- Scalar and array assignment
- Unary and binary arithmetic
- Boolean logic and comparisons
- Conditional execution (`IF / ELIF / ELSE`).
- Range-based `FOR` loops
- Functions and calls
- Input (`GET`) and output (`PRINT`, `ERROR`)
- Explicit program termination (`EXIT`, `ABORT`)

---
