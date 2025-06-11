## What is this project?

This project uses an antlr4 grammar, an MLIR builder, and MLIR lowering to LLVM IR.

It implements a toy calculator language that supports a few primitive linguistic elements:

* a DECLARE operation (implicit double type)
* Fixed size integer declaration operations (INT8, INT16, INT32, INT64)
* Floating point declaration operations (FLOAT32, FLOAT64)
* Boolean declaration operation (BOOL)
* a PRINT operation,
* single line comments,
* a EXIT operation,
* boolean, integer and floating point constants.
* an ASSIGNMENT operator `(=)` with unary `(+,-)` and binary operators `(+,-,*,/)`.
* DWARF instrumentation support, sufficient to for line stepping, breakpoints, continue, and variable inspection (and probably modification: untested.)
* comparison operators (<, <=, EQ, NE) yielding BOOL values.  These work for any combinations of floating and integer types (including BOOL.)
* integer bitwise operators (OR, AND, XOR).  These only for for integer types (including BOOL.)
* a NOT operator, yielding BOOL.
* array declarations, using any of the types above.  STRING is available as an alias for INT8 array.  string literal ASSIGN and PRINT implemented.

Computations occur in assignment operations, and any types are first promoted to the type of the variable.
This means that 'x = 1.99 + 2.99' has the value 3, if x is an integer variable.

The goal was to understand the MLIR ecosystem.
I'd seen MLIR in action in prototype projects at work, but didn't get to play with it first hand.
Overall, it just makes sense to use a structured mechanism to generate an AST equivalent, instead of handcoding the parser to AST stage of the compiler.
This can include built in semantic checking (not part of this toy compiler yet), and avoid the evil of premature lowering (as described by the MLIR white paper.)

AI tools (Grok and ChatGPT) were used to generate some of the initial framework for this project.
As of the time of this writing (Apr 2025), considerable effort is required to keep both Grok and ChatGPT from halucinating MLIR or LLVM APIs that don't exist,
but the tools were invaluable for getting things started.

As an example of the pain of working with AI tools, here's a trivial example: I asked Grok to add comments to my grammar and fix the indenting, but it took 20 minutes to coerse it to use the grammar that I asked it to use (as it claims the ability to read internet content), but it kept making stuff up and injecting changes to the semantics and making changing grammar element name changes that would have broken my listener class.

I'd like to add enough language elements to the project to make it interesting, and now that I have the basic framework, I should be able to
do that without bothering with AI tools that can be more work to use than just doing it yourself.

## Interesting files

* Toy.g4            -- The Antlr4 grammar for the calculator.
* src/driver.cpp    -- This is the compiler driver, handles command line options, opens output files, and orchestrates all the lower level actions (parse tree walk + MLIR builder, lowering to LLVM-IR, assembly printer, and calls the linker.)
* src/calculator.td -- This is the MLIR dialect that defines the compiler eye view of all the grammar elements.
* src/parser.cpp    -- This is the Antlr4 parse tree walker and the MLIR builder.
* src/lowering.cpp  -- LLVM-IR lowering classes.
* prototypes/simplest.cpp  -- A MWE w/ working DWARF instrumentation.  Just emits LLVM-IR and has no assembly printing pass like the toy compiler.
* prototypes/hibye.cpp  -- A MWE w/ working DWARF instrumentation.  This one emits LLVM-IR for a program that includes an IF statement.
* `samples/*.toy` and `bin/testit` -- sample programs and a rudimentary regression test suite based on them.
* bin/build, bin/rebuild -- build scripts (first runs cmake and ninja and sets compiler override if required), second just ninja with some teeing and grepping.

## Command line options

* --output-directory
* --emit-llvm
* --emit-mlir
* --debug (built in MLIR option.)
* -debug-only=toy-driver
* -debug-only=toy-lowering
* --debug-mlir
* -g (show MLIR location info in the dump, and lowered LLVM-IR.)
* -O[0123] -- the usual.
* --stdout.  MLIR and LLVM-IR output to stdout instead of to files.
* --no-emit-object
* -c (compile only, and don't link.)

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
sudo dnf -y install antlr4-runtime antlr4 antlr4-cpp-runtime antlr4-cpp-runtime-devel cmake clang-tools-extra g++ ninja cscope
```

### Building MLIR

On both ubuntu and fedora, I needed a custom build of llvm/mlir, as I didn't find a package that had the MLIR tablegen files.
As it turned out, a custom llvm/mlir build was also required to specifically enable rtti, as altlr4 uses `dynamic_cast<>`.
The -fno-rtti that is required by default to avoid typeinfo symbol link errors, explicitly breaks the antlr4 header files.
That could be avoided if I separated out the antlr4 listener class from the MLIR builder, but the MLIR builder effectively
provides an AST, which I don't need to build myself if I do the builder directly from the listener.

Question: Does the llvm-project has a generic lexer/parser, or does clang/flang/anything-else roll their own?
Having used antlr4 for previous prototyping, also generating a C++ listener, it made sense to me to use what I knew.

See `bin/buildllvm` for how I built and deployed the llvm+mlir installation used for this project.

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

LLVM uses it's own internal dynamic\_cast<> mechanism, so many types appear opaque.  Example:

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


## Build timings:

Fedora 42 ; antlr4-4.13.2
```
real    0m2.034s
user    0m2.654s
sys     0m0.688s
```

Windows-11 w/ WSL2 ubuntu-24 ; antlr4 4.10 (same machine as above, a dual boot windows/fedora system.)
```
real    0m42.288s
user    1m17.337s
sys     0m8.481s
```

Raspberry PI (ubuntu) ; antlr4 4.9.2
```
real    0m54.584s
user    2m9.977s
sys     0m9.730s
```

Interesting that the little PI is almost as fast as the WSL2 ubuntu instance.  Not much justification to keep Windows booted as the primary OS.  Wonder how a Linux VM on Windows would fare compared to WSL2?

