## What is this project?

This project uses an antlr4 grammar, an MLIR builder, and MLIR lowering to LLVM IR.

It implements a toy calculator language that supports a few primitive linguistic elements:

* a DECLARE operation,
* an ASSIGNMENT operator (=) with unary (+,-) and binary operators (+,-,*,/), and
* a PRINT operation.

Integer constants are allowed in the assignment, but compuations are (currently) floating point.

The goal was to understand the MLIR ecosystem.
I'd seen MLIR in action in prototype projects at work, but didn't get to play with it first hand.
Overall, it just makes sense to use a structured mechanism to generate an AST equivalent, with built in semantic checking as desired, instead of handcoding the parser to AST stage of the compiler.

AI tools (Grok and ChatGPT) were used to generate some of the initial framework for this project.
As of the time of this writing (Apr 2025), considerable effort is required to keep both Grok and ChatGPT from halucinating MLIR or LLVM APIs that don't exist,
but the tools were invaluable for getting things started.

I'd like to add enough language elements to the project to make it interesting, and now that I have the basic framework, I should be able to
do that without bothering with AI tools that can be more work to use than just doing it yourself.

Example: samples/foo.toy

```
DCL x;
x = 3;
     PRINT x;
```

This is not the simplest program, which would just be a DECLARE, but the simplest non-trivial program that generates enough IR to be interesting.

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
#loc1 = loc("samples/foo.toy":1:1)
#loc2 = loc("samples/foo.toy":2:5)
#loc3 = loc("samples/foo.toy":2:1)
#loc4 = loc("samples/foo.toy":3:6)
```

As currently coded, the LLVM IR lowering is broken, and gets as far as:

```
>./build/toycalculator  samples/foo.toy --emit-llvm --debug
block with no terminator, has "llvm.call"(%4) <{CConv = #llvm.cconv<ccc>, TailCallKind = #llvm.tailcallkind<none>, callee = @__toy_print, fastmathFlags = #llvm.fastmath<none>, op_bundle_sizes = array<i32>, operandSegmentSizes = array<i32: 1, 0>}> : (f64) -> ()
mlir-asm-printer: 'builtin.module' failed to verify and will be printed in generic form
"builtin.module"() ({
  "llvm.func"() <{CConv = #llvm.cconv<ccc>, function_type = !llvm.func<i32 ()>, linkage = #llvm.linkage<external>, sym_name = "__toy_main", visibility_ = 0 : i64}> ({
    %0 = "llvm.mlir.constant"() <{value = 1 : i64}> : () -> i64
    %1 = "llvm.alloca"(%0) <{alignment = 8 : i64, elem_type = f64}> : (i64) -> !llvm.ptr
    %2 = "llvm.mlir.constant"() <{value = 3 : i64}> : () -> i64
    %3 = "llvm.sitofp"(%2) : (i64) -> f64
    "llvm.store"(%3, %1) <{ordering = 0 : i64}> : (f64, !llvm.ptr) -> ()
    %4 = "llvm.load"(%1) <{ordering = 0 : i64}> : (!llvm.ptr) -> f64
    "llvm.func"() <{CConv = #llvm.cconv<ccc>, function_type = !llvm.func<void (f64)>, linkage = #llvm.linkage<external>, sym_name = "__toy_print", sym_visibility = "private", visibility_ = 0 : i64}> ({
    }) : () -> ()
    "llvm.call"(%4) <{CConv = #llvm.cconv<ccc>, TailCallKind = #llvm.tailcallkind<none>, callee = @__toy_print, fastmathFlags = #llvm.fastmath<none>, op_bundle_sizes = array<i32>, operandSegmentSizes = array<i32: 1, 0>}> : (f64) -> ()
  ^bb1:  // no predecessors
    %5 = "llvm.mlir.constant"() <{value = 0 : i32}> : () -> i32
    "llvm.return"(%5) : (i32) -> ()
  }) : () -> ()
}) : () -> ()
...
loc("samples/foo.toy":3:6): error: block with no terminator, has "llvm.call"(%4) <{CConv = #llvm.cconv<ccc>, TailCallKind = #llvm.tailcallkind<none>, callee = @__toy_print, fastmathFlags = #llvm.fastmath<none>, op_bundle_sizes = array<i32>, operandSegmentSizes = array<i32: 1, 0>}> : (f64) -> ()
FATAL ERROR: LLVM lowering failed
```

I don't like the llvm.func for PRINT runtime method embedded in the codegen for the fake "main" function for the program.
It would be good to have a pass that finds all the global symbols and emits the LLVM IR for those upfront.  Alternatively, the parse listener could collect all the global symbols (right now just PRINT, serialized as __toy_print), and generate an MLIR element for all such symbols that could be processed in a pass before any other codegen.
It actually makes things difficult not to have a notion of functions (esp. one for main), and adding that would simplify lowering (and make the language have some utility.)

## TODO

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

