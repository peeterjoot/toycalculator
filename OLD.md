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
                         << "Lowering silly.program: " << *op << '\n' << loc << '\n' );
```

In particular, the dump() function can be used for many mlir objects.  That coupled with `--debug` in the driver is the primary debug mechanism that I have used developing this compiler.

## Experimenting with symbol tables.

Now using symbol tables instead of hashing in parser/builder, but not in lowering.  An attempt to do so can be found in the branch `peeter/old/symbol-table-tryII`.

Everything in that branch was merged to master in one big commit that wipes out all the false starts in that branch (that merge also includes the `peeter/old/if-else` branch.)
