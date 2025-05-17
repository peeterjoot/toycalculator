## What is this project?

This project uses an antlr4 grammar, an MLIR builder, and MLIR lowering to LLVM IR.

It implements a toy calculator language that supports a few primitive linguistic elements:

* a DECLARE operation,
* a PRINT operation,
* single line comments,
* a EXIT operation,
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

## Interesting files

* Toy.g4            -- The Antlr4 grammar for the calculator.
* src/driver.cpp    -- This is the compiler driver, handles command line options, opens output files, and orchestrates all the lower level actions (parse tree walk + MLIR builder, lowering to LLVM-IR, and assembly printer).
* src/calculator.td -- This is the MLIR dialect that defines the compiler eye view of all the grammar elements.
* src/parser.cpp    -- This is the Antlr4 parse tree walker and the MLIR builder.
* src/lowering.cpp  -- LLVM-IR lowering classes.

## Command line options

* --output-directory
* --emit-llvm
* --emit-mlir
* --debug (built in MLIR option.)
* -debug-only=toy-driver
* -debug-only=toy-lowering
* --debug-mlir
* -g (show MLIR location info in the dump, and generate DWARF metadata in the lowered LLVM-IR.)
* -O[0123] -- the usual.
* --stdout.  MLIR and LLVM-IR output to stdout instead of to files.
* --no-emit-object.

## TODO

Basic language constructs to make things more interesting:

* tests for all the type conversions.
* Lots of cut and paste duplication for type conversion in lowering.cpp -- split out into helper functions.
* unary.toy: if x = -x, is changed to x = 0 - x, the program doesn't compile.
* EXIT: enforce i8 return type in the MLIR layer (i.e.: actual UNIX shell semantics.) -- currently set to i32 return.
* Boolean operators.
* Implement IF/WHILE/DO/BREAK/CONTINUE statements.
* More complicated expressions.
* CAST operators.
* Allow EXIT at more than the end of program (currently enforced in the grammar.)
* Don't have any traits defined for my MLIR operations (initially caused compile errors, and I just commented-out or omitted after that.)

Trickier, but fun stuff:
* LLVM IR lowering doesn't produce DWARF instrumentation matching the location info?
* Implement a JIT so that the "language" has the capability of a static compilation mode, as well as interpretted.

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

## OLD Examples (<= tag: V0):

These are MLIR samples that applied to the pre-symboltable version of the code:

1. samples/empty.toy

```
// This should be allowed by the grammar.
```

The MLIR for this program used to be:

```
> ../build/toycalculator empty.toy  -g
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
#loc = loc("../samples/empty.toy":1:1)
#loc1 = loc("../samples/empty.toy":2:1)
```

FIXME: where did location 2:1 come from in `#loc1` above for the return statement?

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
> ../build/toycalculator dcl.toy  -g
"builtin.module"() ({
  "toy.program"() ({
    %0 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<f64> loc(#loc1)
    "toy.declare"() <{name = "x"}> : () -> () loc(#loc1)
    "toy.return"() : () -> () loc(#loc1)
  }) : () -> () loc(#loc1)
}) : () -> () loc(#loc)
#loc = loc("../samples/dcl.toy":1:1)
#loc1 = loc("../samples/dcl.toy":1:1)
```

3. samples/foo.toy

This is the simplest non-trivial program that generates enough IR to be interesting.

```
DCL x;
x = 3;
// This indenting is to test location generation, and to verify that the resulting columnar position is right.
     PRINT x;
```

Here is the MLIR for the code above (for an older version of this project, now toy.unary is replaced with either toy.negate, or nothing):

```
> ./build/toycalculator  samples/foo.toy  -g
"builtin.module"() ({
  "toy.program"() ({
    %0 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<f64> loc(#loc1)
    "toy.declare"() <{name = "x"}> : () -> () loc(#loc1)
    %1 = "arith.constant"() <{value = 3 : i64}> : () -> i64 loc(#loc2)
    %2 = "toy.unary"(%1) <{op = "+"}> : (i64) -> f64 loc(#loc2)
    "memref.store"(%2, %0) : (f64, memref<f64>) -> () loc(#loc2)
    "toy.assign"(%2) <{name = "x"}> : (f64) -> () loc(#loc3)
    "toy.print"(%0) : (memref<f64>) -> () loc(#loc4)
    "toy.return"() : () -> () loc(#loc1)
  }) : () -> () loc(#loc1)
}) : () -> () loc(#loc)
#loc = loc("../samples/foo.toy":1:1)
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

4. samples/test.toy

```
DCL x;
DCL y;
x = 5 + 3;
y = x * 2;
PRINT x;
PRINT y;
```

Note: The cut and paste above no longer matches the repo, as samples/test.toy now uses 3.14E0 instead of 3, since I added floating point constants to the grammar and parse listener.

The LL lowering results look pretty nice:
```
; ModuleID = 'test.toy'
source_filename = "test.toy"

declare void @__toy_print(double)

define i32 @main() {
  %1 = alloca double, i64 1, align 8
  %2 = alloca double, i64 1, align 8
  store double 8.000000e+00, ptr %1, align 8
  %3 = load double, ptr %1, align 8
  %4 = fmul double %3, 2.000000e+00
  store double %4, ptr %2, align 8
  %5 = load double, ptr %1, align 8
  call void @__toy_print(double %5)
  %6 = load double, ptr %2, align 8
  call void @__toy_print(double %6)
  ret i32 0
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
```

The assembler printer (with -O 2) reduces all the double operations to constant lookups:
```
> objdump -d samples/out/test --no-show-raw-insn | grep -A8 '<main>:'
0000000000400470 <main>:
  400470:       push   %rax
  400471:       movsd  0xd27(%rip),%xmm0        # 4011a0 <__dso_handle+0x8>
  400479:       call   400370 <__toy_print@plt>
  40047e:       movsd  0xd22(%rip),%xmm0        # 4011a8 <__dso_handle+0x10>
  400486:       call   400370 <__toy_print@plt>
  40048b:       xor    %eax,%eax
  40048d:       pop    %rcx
  40048e:       ret
```

## Experimenting with symbol tables.

It made sense to me to emit MLIR that encoded symbol attributes.  I tried:

```
def Toy_ProgramOp : Op<Toy_Dialect, "program"> {
  let summary = "Program operation";
  let arguments = (ins);
  let results = (outs);
  let regions = (region AnyRegion:$body);
  let traits = [AutomaticAllocationScope, SymbolTable];
}

def Toy_DeclareOp : Op<Toy_Dialect, "declare"> {
  let summary = "Declare a variable in the symbol table";
  let arguments = (ins TypeAttr:$type);
  let results = (outs);
  let traits = [Symbol];
}

def Toy_AssignOp : Op<Toy_Dialect, "assign"> {
  let summary = "Assign a value to a variable by symbol";
  let arguments = (ins AnyType:$value, SymbolRefAttr:$name);
  let results = (outs);
}

def Toy_LoadOp : Op<Toy_Dialect, "load"> {
  let summary = "Load a variable’s value by symbol";
  let arguments = (ins SymbolRefAttr:$name);
  let results = (outs AnyType:$value);
}
```

This worked out well in the builder, where I could create a new symbol using:

```
        auto dcl = builder.create<toy::DeclareOp>( loc, mlir::TypeAttr::get( ty ) );
        dcl->setAttr( "sym_name", builder.getStringAttr( varName ) );
```

and also reference it
```
            if ( auto *symbolOp = mlir::SymbolTable::lookupSymbolIn( programOp, varName ) )
            {
                if ( auto declareOp = llvm::dyn_cast<toy::DeclareOp>( symbolOp ) )
                {
                    mlir::Type varType = declareOp.getTypeAttr().getValue();
                    auto sref = mlir::SymbolRefAttr::get( builder.getContext(), varName );
                    value = builder.create<toy::LoadOp>( loc, varType, sref );
                    ...
```

but I couldn't figure out how to get the lowering to work.  Here's an example program:

```
BOOL i1;
i1 = TRUE;
BOOL i2;
i2 = i1;
```

and the corresponding IR
```
module {
  "toy.program"() ({
    "toy.declare"() <{type = i1}> {sym_name = "i1"} : () -> ()
    %true = arith.constant true
    "toy.assign"(%true) <{name = @i1}> : (i1) -> ()
    "toy.declare"() <{type = i1}> {sym_name = "i2"} : () -> ()
    %0 = "toy.load"() <{name = @i1}> : () -> i1
    "toy.assign"(%0) <{name = @i2}> : (i1) -> ()
    toy.exit
  }) : () -> ()
}
```

My ProgramOpLowering kicked in first, which effectively deletes the symbol table.  After that declare lowering gets screwed up, resulting in trace (`--debug`) output like:

```
Trying to match "{anonymous}::DeclareOpLowering"
Lowering toy.declare: 'toy.declare' op symbol's parent must have the SymbolTable trait
mlir-asm-printer: 'llvm.func' failed to verify and will be printed in generic form
```

I suspect that I would need to essentially "move" that symbol table to the new parent block for the declare, just like the basic block itself was moved.  This was my last failed attempt at the declare lowering:

```
    class DeclareOpLowering : public OpRewritePattern<toy::DeclareOp>
    {
        using OpRewritePattern::OpRewritePattern;

       public:
        DeclareOpLowering( loweringContext& lState, MLIRContext* context )
            : OpRewritePattern( context ), lState( lState )
        {
        }

        LogicalResult matchAndRewrite( toy::DeclareOp dcl, PatternRewriter& rewriter ) const override
        {
            auto loc = dcl.getLoc();

            // example:
            //
            // "toy.declare"() <{type = i1}> {sym_name = "i1"} : () -> ()
            LLVM_DEBUG( llvm::dbgs() << "Lowering toy.declare: " << dcl << '\n' );
            rewriter.setInsertionPoint( dcl );
            rewriter.eraseOp( dcl );

            auto nameAttr = mlir::dyn_cast<mlir::StringAttr>( dcl->getAttr( "sym_name" ) );
            if ( !nameAttr )
                return rewriter.notifyMatchFailure( dcl, "expected 'sym_name' to be a StringAttr" );

            auto varName = nameAttr.getValue();
            auto elemType = dcl.getType();
            int64_t numElements = 1;    // scalar only for now.

            unsigned elemSizeInBits = elemType.getIntOrFloatBitWidth();
            unsigned elemSizeInBytes = ( elemSizeInBits + 7 ) / 8;
            int64_t totalSizeInBytes = numElements * elemSizeInBytes;


            auto ptrType = LLVM::LLVMPointerType::get( rewriter.getContext() );
            if ( !lState.onei64a )
            {
                lState.onei64 =
                    rewriter.create<LLVM::ConstantOp>( loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr( 1 ) );
                lState.onei64a = true;
            }
            auto newAllocaOp =
                rewriter.create<LLVM::AllocaOp>( loc, ptrType, elemType, lState.onei64, totalSizeInBytes );

            lState.symbolToAlloca[nameAttr] = newAllocaOp;

            rewriter.getInsertionBlock()->dump();

            return success();
        }

       private:
        loweringContext& lState;
    };
```

This has the appearance of working as I see the erase in the trace output, but it seems to get rolled back (as seen in the final dump).  I also tried mutating versions where I deleted the symbol references from the declareop.  I'm sure that's the wrong approach too.

The initial exploration related to the attempt to use symbol tables (as well as generalize the supported types) can be found in the branch: `types_and_symbol_table`.  When merged to master (tag: V1), I squashed all those commits, obliterating the symbol table experimentation.
