## Motivation

The goal of this project was to gain concrete, hands-on experience with the MLIR ecosystem.
It uses an ANTLR4 grammar, an MLIR builder, and MLIR lowering to LLVM IR, incorporating a custom dialect (toy) along with several existing MLIR dialects (scf, arith, memref, etc.).

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

Initially, I used MLIR to build a simple symbolic calculator that supported double-like variable declarations, assignments, unary and binary arithmetic operations, and output display.

As noted earlier, the primary goal of the project was not calculation itself, but to gain concrete, hands-on experience with the MLIR ecosystem.

That initial implementation has evolved into a toy language and its compiler, which now supports the following features:

* A `DECLARE` operation (implicit double type).
* Fixed-size integer declarations (`INT8`, `INT16`, `INT32`, `INT64`).
* Floating-point declarations (`FLOAT32`, `FLOAT64`).
* Boolean declaration (`BOOL`).
* A `PRINT` operation for printing to standard output.
* A `GET` operation for reading from standard input.
* Single-line comments.
* An `EXIT` operation.
* Boolean, integer, and floating-point constants, along with expression evaluation.
* An ASSIGNMENT operator (`=`) with unary (`+`, `-`) and binary operators (`+`, `-`, `*`, `/`).
* DWARF instrumentation sufficient for line stepping, breakpoints, continue, and variable inspection (variable modification is likely supported but untested).
* Comparison operators (`<`, `<=`, `==`, `!=`) yielding `BOOL` values. These work across any combinations of floating-point and integer types (including `BOOL`).
* Integer bitwise operators (`OR`, `AND`, `XOR`), applicable only to integer types (including `BOOL`).
* A `NOT` operator yielding `BOOL`.
* Array support, including declaration, assignment, printing, returning, exiting, and element access.
* A `STRING` type as an alias for `INT8` arrays, with string literal assignment and `PRINT` implemented.
* User-defined functions. Calls use the form `CALL function_name(p1, p2)` or with assignment `x = CALL function_name(p1, p2)`. Declarations use: `FUNCTION foo(type name, type name, ...) : RETURN-type { ... ; RETURN v; };` (where : `RETURN-type` is optional).
* `IF`/`ELSE` statement support. The grammar includes an `ELIF` construct, but it is not yet implemented. Logical operators (`AND`, `OR`, `XOR`) are not supported in predicates (only comparisons like `<`, `>`, `<=`, `>=`, etc.). Complex predicates (e.g., `(a < b) AND (c < d)`) are not supported. Nested `IF`s are untested and may or may not work.
* A `FOR` loop (supporting start, end, and step-size params, and requiring a previously defined variable for the loop induction variable.)

There is lots of room to add add further language elements to make the compiler and language more interesting.  Some ideas for improvements (as well as bug fixes) can be found in TODO.md

## Language Quirks.

* Like scripted languages, there is an implicit `main` in this toy language.
* Functions can be defined anywhere, but must be defined before use.
* Computations occur in assignment operations, and any types are first promoted to the type of the variable.
This means that `x = 1.99 + 2.99` has the value `3`, if `x` is an integer variable, but `4.98` if x is a `FLOAT32` or `FLOAT64`.
* The `EXIT` statement currently has to be at the end of the program.
`EXIT` without a numeric value is equivalent to `EXIT 0`, as is a program with no explicit `EXIT`.
* The RETURN statement has to be at the end of a function.  It is currently mandatory.
* See TODO.md for a long list of nice to have features that I haven't gotten around to yet, and may never.
* `GET` into a `BOOL` value will abort if the value isn't one of 0, or 1.  This is inconsistent with assignment to a BOOL variable, which will truncate and not raise a runtime error.

## On the use of AI in this project.

AI tools (Grok and ChatGPT) were used to generate some of the initial framework for this project (April 2025 timeframe.)
At that point in time, considerable effort was required to keep both Grok and ChatGPT from hallucinating MLIR or LLVM APIs that don't exist,
but both of those models were invaluable for getting things started.

As an example of the pain of working with AI tools, here's a trivial example: I asked Grok to add comments to my grammar and fix the indenting, but it took 20 minutes to coerce it to use the grammar that I asked it to use (as it claims the ability to read internet content), but it kept making stuff up and injecting changes to the semantics and making changing grammar element name changes that would have broken my listener class.

## Interesting files

* `Toy.g4`           -- The Antlr4 grammar.
* `src/driver.cpp`   -- This is the compiler driver, handles command line options, opens output files, and orchestrates all the lower level actions (parse tree walk + MLIR builder, lowering to LLVM-IR, assembly printer, and calls the linker.)
* `src/calculator.td` -- This is the MLIR dialect that defines the compiler eye view of all the grammar elements.
* `src/parser.cpp`   -- This is the Antlr4 parse tree walker and the MLIR builder.
* `src/lowering.cpp` -- LLVM-IR lowering classes.
* `prototypes/simplest.cpp`  -- A MWE w/ working DWARF instrumentation.  Just emits LLVM-IR and has no assembly printing pass like the toy compiler.
* `prototypes/hibye.cpp` -- A MWE w/ working DWARF instrumentation.  This one emits LLVM-IR for a program that includes an `IF` statement.
* `samples/*.toy` and `bin/testit` -- sample programs and a rudimentary regression test suite based on them.
* `bin/build`, `bin/rebuild` -- build scripts (first runs cmake and ninja and sets compiler override if required), second just ninja with some teeing and grepping.

## Command line options

* `--output-directory`
* `--emit-llvm`
* `--emit-mlir`
* `--debug` (built in MLIR option.)
* `-debug-only=toy-driver`
* `-debug-only=toy-lowering`
* `--debug-mlir`
* `-g` (show MLIR location info in the dump, and lowered LLVM-IR.)
* `-O[0123]` -- the usual.
* `--stdout`.  MLIR and LLVM-IR output to stdout instead of to files.
* `--no-emit-object`
* `-c` (compile only, and don't link.)

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
sudo dnf -y install antlr4-runtime antlr4 antlr4-cpp-runtime antlr4-cpp-runtime-devel cmake clang-tools-extra g++ ninja cscope clang++ ccache
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

## Debugging

### Peeking into LLVM object internals

LLVM uses it's own internal `dynamic_cast<>` mechanism, so many types appear opaque.  Example:

```
(gdb) p loc
$2 = {impl = {<mlir::Attribute> = {impl = 0x5528d8}, <No data fields>}}
```

If we happen to know the real underlying type, we can cast the impl part of the object

```
(gdb) p *(mlir::FileLineColLoc*)loc.impl
$3 = {<mlir::FileLineColRange> = {<mlir::detail::StorageUserBase<mlir::FileLineColRange, mlir::LocationAttr, mlir::detail::FileLineColRangeAttrStorage, mlir::detail::AttributeUniquer, mlir::AttributeTrait::IsLocation>> = {<mlir::LocationAttr> = {<mlir::Attribute> = {
          impl = 0x539330}, <No data fields>}, <mlir::AttributeTrait::IsLocation<mlir::FileLineColRange>> = {<mlir::detail::StorageUserTraitBase<mlir::FileLineColRange, mlir::AttributeTrait::IsLocation>> = {<No data fields>}, <No data fields>}, <No data fields>}, <No data fields>}, <No data fields>}
```

but that may not be any more illuminating.  Old fashioned printf style debugging does work:

```
             LLVM_DEBUG( llvm::dbgs()
                         << "Lowering toy.program: " << *op << '\n' << loc << '\n' );
```

In particular, the dump() function can be used for many mlir objects.  That coupled with `--debug` in the driver is the primary debug mechanism that I have used developing this compiler.

## Experimenting with symbol tables.

Now using symbol tables instead of hashing in parser/builder, but not in lowering.  An attempt to do so can be found in the branch `peeter/old/symbol-table-tryII`.

Everything in that branch was merged to master in one big commit that wipes out all the false starts in that branch (that merge also includes the `peeter/old/if-else` branch.)
