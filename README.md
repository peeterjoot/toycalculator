## What is this project?

This project uses an antlr4 grammar, an MLIR builder, and MLIR lowering to LLVM IR.

It implements a toy calculator language that supports a few primitive linguistic elements:

* a DECLARE operation,
* a PRINT operation,
* single line comments,
* a RETURN operation,
* an ASSIGNMENT operator (=) with unary (+,-) and binary operators (+,-,*,/).

Integer constants are allowed in the assignment, but compuations are (currently) floating point.

The goal was to understand the MLIR ecosystem.
I'd seen MLIR in action in prototype projects at work, but didn't get to play with it first hand.
Overall, it just makes sense to use a structured mechanism to generate an AST equivalent, with built in semantic checking as desired, instead of handcoding the parser to AST stage of the compiler.

AI tools (Grok and ChatGPT) were used to generate some of the initial framework for this project.
As of the time of this writing (Apr 2025), considerable effort is required to keep both Grok and ChatGPT from halucinating MLIR or LLVM APIs that don't exist,
but the tools were invaluable for getting things started.

As an example of the pain of working with AI tools, here's a trivial example: I asked Grok to add comments to my grammar and fix the indenting, but it took 20 minutes to coerse it to use the grammar that I asked it to use (as it claims the ability to read internet content), but it kept making stuff up and injecting changes to the semantics and making changing grammar element name changes that would have broken my listener class.

I'd like to add enough language elements to the project to make it interesting, and now that I have the basic framework, I should be able to
do that without bothering with AI tools that can be more work to use than just doing it yourself.

Examples:

1. samples/empty.toy

```
// This should be allowed by the grammar.
```

The MLIR for this program used to be:

```
> ../build/toycalculator empty.toy  --location
"builtin.module"() ({
  "toy.program"() ({
  ^bb0:
  }) : () -> () loc(#loc1)
}) : () -> () loc(#loc)
#loc = loc(unknown)
#loc1 = loc("empty.toy":2:1)
```

Where the '^bb0:' is the MLIR dump representation of an empty basic block.
To help fix this, I introduced a return statement grammar element, and force an implicit return onto the program's BB in the builder, if return was not specified.

```
"builtin.module"() ({
  "toy.program"() ({
    "toy.return"() : () -> () loc(#loc1)
  }) : () -> () loc(#loc1)
}) : () -> () loc(#loc)
#loc = loc(unknown)
#loc1 = loc("../samples/empty.toy":2:1)
```

(FIXME: where did line two come from in loc1 above for the return statement.  Why isn't that line 1 -- the only line in the file?  Also, it would look prettier to have unknown map to line of of the .toy file -- although does that matter since we strip the builtin.module in lowering?)

With that return implemented, and a bunch of lowering tweaks (i.e.: so we don't crash in lowering ProgramOp and ReturnOp), we can now lower this to LLVM-IR:

```
; ModuleID = '../samples/empty.toy'
source_filename = "../samples/empty.toy"

define i32 @main() {
  ret i32 0
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
```

FIXME: Whatever this "Debug Info" stuff is, it doesn't appear to correspond to the MLIR location info.

2.  samples/dcl.toy

```
DCL x; // The simplest non-empty program.
```

Results in MLIR like:

```
> ../build/toycalculator dcl.toy  --location
"builtin.module"() ({
  "toy.program"() ({
    %0 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<f64> loc(#loc1)
    "toy.declare"() <{name = "x"}> : () -> () loc(#loc1)
  }) : () -> () loc(#loc1)
}) : () -> () loc(#loc)
#loc = loc(unknown)
#loc1 = loc("dcl.toy":1:1)
```

3. samples/foo.toy

This is the simplest non-trivial program that generates enough IR to be interesting.

```
DCL x;
x = 3;
// This indenting is to test location generation, and to verify that the resulting columnar position is right.
     PRINT x;
```

Here is the MLIR for the code above:

```
> ./build/toycalculator  samples/foo.toy  --location
"builtin.module"() ({
  "toy.program"() ({
    %0 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<f64> loc(#loc1)
    "toy.declare"() <{name = "x"}> : () -> () loc(#loc1)
    %1 = "arith.constant"() <{value = 3 : i64}> : () -> i64 loc(#loc2)
    %2 = "toy.unary"(%1) <{op = "+"}> : (i64) -> f64 loc(#loc2)
    "memref.store"(%2, %0) : (f64, memref<f64>) -> () loc(#loc2)
    "toy.assign"(%2) <{name = "x"}> : (f64) -> () loc(#loc3)
    "toy.print"(%0) : (memref<f64>) -> () loc(#loc4)
  }) : () -> () loc(#loc1)
}) : () -> () loc(#loc)
#loc = loc(unknown)
#loc1 = loc("../samples/foo.toy":1:1)
#loc2 = loc("../samples/foo.toy":2:5)
#loc3 = loc("../samples/foo.toy":2:1)
#loc4 = loc("../samples/foo.toy":4:6)
```

The LLVM IR lowering looks like:

```
; ModuleID = 'foo.toy'
source_filename = "foo.toy"

declare void @__toy_print(double)

define i32 @main() {
  %1 = alloca double, i64 1, align 8
  store double 3.000000e+00, ptr %1, align 8
  %2 = load double, ptr %1, align 8
  call void @__toy_print(double %2)
  ret i32 0
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
```

Notice that the !dbg location info is MIA, despite it having been in the MLIR dump.

However, we can actually link and run the code without error:

```
> ../build/toycalculator foo.toy --output-directory out
Generated object file: out/foo.o

> clang -o foo ../samples/out/foo.o -L ../build -l toy_runtime -Wl,-rpath,`pwd`/../build

> ./foo
3.000000
```

## Command line options

* --output-directory
* --emit-llvm
* --emit-mlir
* --debug (mlir default option.)
* --debug-mlir
* --location (show MLIR location info in the dump.)
* -O[0123] -- the usual.

## TODO

* Add support for a numeric and symbol value for the RETURN statement (grammar, listener, builder, lowering.)
* LLVM IR lowering.  This is in progress.
* Floating point constants (will touch the grammar, builder and lowering.)
* Types: fixed size integers and maybe floating point types of different sizes (not just double equivialent.)
* Function calls (to more than the single PRINT runtime function.)
* Implement a JIT so that the "language" has the capability of a static compilation mode, as well as interpretted.

## Building

### anltlr4 setup (ubuntu)

```
sudo apt-get install libantlr4-runtime-dev
sudo apt-get install antlr4
wget https://www.antlr.org/download/antlr-4.10-complete.jar
```

This assumes that the antlr4 runtime, after installation, is 4.10 -- if not, change appropriately (but CMakeLists.txt will also need to be updated.)

I had issues where the installed runtime didn't exactly match the generator.  I forget which system I saw that on.  That's why I had the separate jar download, but would have to verify which machine actualy required that.  Should abstract out the antlr4 invocation, instead of having complicated Cmake logic.

### anltlr4 setup (Fedora)

```
sudo dnf -y install antlr4-runtime antlr4 antlr4-cpp-runtime antlr4-cpp-runtime-devel
wget https://www.antlr.org/download/antlr-4.10-complete.jar
```

Like above, this assumes that the antlr4 runtime is 4.10.

### Building MLIR

On both ubuntu and fedora, I needed a custom build of llvm/mlir, as I didn't find a package that had the MLIR tablegen files.
As it turned out, a custom llvm/mlir build was also required to specifically enable rtti, as altlr4 uses dynamic_cast<>.
The -fno-rtti that is required by default to avoid typeinfo symbol link errors, explicitly breaks the antlr4 header files.
I'm not actually sure if the llvm-project has a generic lexer/parser, or if those are all language specific.
Having used antlr4 for previous prototyping, also generating a C++ listener, it made sense to me to use what I knew.

See bin/buildllvm for how I built and deployed the llvm+mlir installation used for this project.

### Building the project.

```
. ./bin/env
build.sh
```

The build script current assumes that I'm the one building it, and is likely not sufficiently general for other people to use, and will surely break as I upgrade the systems I attempt to build it on.

Linux-only is assumed.

Depending on what I currently have booted, this has been on a mix of:

* Fedora 42/X64 (on a dual boot windows-11/Linux laptop)
* WSL ubuntu 24/X64 (windows side, same laptop.)
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

