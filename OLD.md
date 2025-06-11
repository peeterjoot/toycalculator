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
