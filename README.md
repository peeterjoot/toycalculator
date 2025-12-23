## Motivation

The point of this project was to get some concrete first hand experience with the MLIR ecosystem.
This project uses an antlr4 grammar, an MLIR builder, and MLIR lowering to LLVM IR, using a custom dialect (toy) and a few existing MLIR dialects (scf, arith, memref, ...)

I'd seen MLIR in action in a prototype project at work, but didn't get to play with it first hand.
It appeared to me that MLIR provided a structured mechanism that avoided the requirement to hand craft an AST, allowing for built in semantic checking, and a robust description of a set of sources that could be used for transformations.
The MLIR white paper describes the evils of premature lowering.
It's my recollection that this was primarily with respect to lowering to object code was in mind.
However, a good high level representation of a program can also facilitate high level transformations, even transformations between languages, and also code query tasks.

I've used the clang AST API to query code, generate code, and make transformations.
Some of those applications were in very large codebases where it is difficult to automate large structural changes.
I've also used proprietary AST walkers in commercial compilers, where infrastructure of that sort can be used to extend the compiler itself, supplying user defined semantics that are appropriate to the customer base.
Such tools are immensely powerful and can be very useful, so having a structured representation of sources and a language has a lot of appeal.
The power of such an approach seems clear, and exciting enough to try out for myself, even if I did not have a work justification to do so.

## What is this project?

Initially, I used MLIR to build a simple symbolic calculator.
This calculator supported `double`-like variable declarations, assignments, arithmetic unary and binary operations, and output display.
As indicated above, the point of the project wasn't really to calculate, but just to get some concrete first hand experience with the MLIR ecosystem for myself.

That initial implementation has morphed into a implementation of a toy language and compiler for that language, which now supports the following linguistic elements:


* a DECLARE operation (implicit double type)
* Fixed size integer declaration operations (INT8, INT16, INT32, INT64)
* Floating point declaration operations (FLOAT32, FLOAT64)
* Boolean declaration operation (BOOL)
* a PRINT operation,
* single line comments,
* a EXIT operation,
* boolean, integer and floating point constants, and expression evaluation.
* an ASSIGNMENT operator `(=)` with unary `(+,-)` and binary operators `(+,-,*,/)`.
* DWARF instrumentation support, sufficient to for line stepping, breakpoints, continue, and variable inspection (and probably modification: untested.)
* comparison operators (<, <=, EQ, NE) yielding BOOL values.  These work for any combinations of floating and integer types (including BOOL.)
* integer bitwise operators (OR, AND, XOR).  These only for for integer types (including BOOL.)
* a NOT operator, yielding BOOL.
* array support, declaration, assignment, print, return, exit and access of an array variable.
* A STRING type is available as an alias for INT8 array.  string literal ASSIGN and PRINT implemented.
* User defined functions.  CALL `function_name( p1, p2)`, or w/ assign; `x = CALL function_name( p1, p2 )`.  Declaration looks like: `FUNCTION foo( type name, type name, ...) : RETURN-type { ... ; RETURN v;};`  (where `:RETURN-TYPE` is optional.)
* IF/ELSE statement support.  The grammar has an ELIF construct too, but that isn't implemented yet.  AND and OR and XOR aren't supported in the predicates yet (just <, >, LE, GT, ...)  Complex predicates are also not supported ((a < b) AND (c < d)).  Nested IFs aren't tested yet, and may or may not work.

I'd like to add enough language elements to the language and compiler to make it interesting.  The biggest missing pieces at this point are the lack of loops, no input mechanism (i.e.: to do something interesting in loops, once implemented.)

## Language Quirks.

* Like scripted languages, there is an implicit `main` in this toy language.
* Functions can be defined anywhere, but must be defined before use.
* Computations occur in assignment operations, and any types are first promoted to the type of the variable.
This means that 'x = 1.99 + 2.99' has the value 3, if x is an integer variable, but 4.98 if x is a FLOAT32 or FLOAT64.
* The EXIT statement currently has to be at the end of the program.  EXIT without a numeric value is equivalent to EXIT 0, as is a program with no explicit EXIT.
* The RETURN statement has to be at the end of a function.  It is currently mandatory.
* See TODO for a long list of nice to have features that I haven't gotten around to yet, and may never.

## On the use of AI in this project.

AI tools (Grok and ChatGPT) were used to generate some of the initial framework for this project (April 2025 timeframe.)
At that point in time, considerable effort was required to keep both Grok and ChatGPT from hallucinating MLIR or LLVM APIs that don't exist,
but both of those models were invaluable for getting things started.

As an example of the pain of working with AI tools, here's a trivial example: I asked Grok to add comments to my grammar and fix the indenting, but it took 20 minutes to coerce it to use the grammar that I asked it to use (as it claims the ability to read internet content), but it kept making stuff up and injecting changes to the semantics and making changing grammar element name changes that would have broken my listener class.

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

In particular, the dump() function can be used for many mlir objects.

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

## Experimenting with symbol tables.

Now using symbol tables instead of hashing in parser/builder, but not in lowering.  An attempt to do so can be found in the branch symbol-table-tryII.

Everything in that branch was merged to master in one big commit that wipes out all the false starts in that branch (that merge also includes the if-else branch.)
