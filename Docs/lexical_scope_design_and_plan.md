# DWARF Lexical Scope Instrumentation — Implementation Plan

## 1. Problem Statement

The silly compiler has a partial, incomplete implementation of DWARF lexical scope
instrumentation. An earlier attempt introduced `DebugScopeOp` and `DILexicalBlockAttr`
attachment for variables declared inside `IF`, `ELIF`, `ELSE`, and `FOR` bodies, but
the implementation is incomplete and buggy:

- FOR induction variables are still scoped to the enclosing `DISubprogram` rather than
  to the loop's lexical block
- The predicate expression of each `IF`/`ELIF` is not scoped to its own `DILexicalBlock`
  (clang produces a separate predicate-scope block distinct from the body block)
- The `DebugScopeOp` approach threads scope information through the value stream as an
  operand to `DebugNameOp`, which is architecturally awkward and does not compose well
  with nested control flow
- gdb line-stepping regressions in loop bodies and after CALL returns are believed to be
  related to incorrect or missing scope attribution on intermediate ops

The reference behavior (from clang's output for equivalent C programs compiled with
`-g -O0`) is:

- Each `IF`/`ELIF` condition expression gets its own `DILexicalBlock` scoped to the
  enclosing function or block
- Each `{ }` body gets its own `DILexicalBlock` scoped to the corresponding predicate
  block
- `ELSE` bodies get a single `DILexicalBlock` scoped directly to the enclosing function
  or block (no predicate block, since there is no predicate expression)
- `FOR` loops get three `DILexicalBlock`s — two for the loop header/induction variable,
  and one for the body
- Variables declared directly in the function body (outside any control flow) are scoped
  to the `DISubprogram` with no intervening lexical block

The plan below is deliberately exploratory. Each step is small and verifiable before
committing to the next. The design questions — in particular how much `FusedLoc` wrapping
is actually needed, and whether scope information is better carried in op locations or in
op attributes — are open and will be answered by the experiments.

---

## 2. Proposed Mechanism (To Be Validated by Experiment)

### 2.1 Structural scope markers in the IR

The fundamental problem is that `DILexicalBlockAttr` objects cannot be constructed
during MLIR generation because they require a `DISubprogramAttr` parent that does not
exist until lowering. The proposed solution is to defer DI construction by emitting
lightweight structural marker ops during MLIR generation:

```
silly.scope_begin { id = <integer>, kind = <scope_kind> }
silly.scope_end   { id = <integer> }
```

These carry no operands or results — they exist purely to mark scope boundaries in
the op sequence at the correct source locations. A pre-lowering pass (or the lowering
itself) then walks the op sequence, maintains a scope stack, constructs the actual
`DILexicalBlockAttr` hierarchy with the `DISubprogramAttr` as root, and either removes
the marker ops or replaces them with the constructed DI attributes.

### 2.2 Attaching scope to op locations

MLIR's translation to LLVM IR reads the scope for each `!DILocation` from the op's
location attribute. The mechanism for carrying a `DIScope` alongside a `FileLineColLoc`
is `FusedLoc`:

```
FusedLoc([FileLineColLoc(file, line, col)], metadata=DILexicalBlockAttr)
```

The column information is preserved inside the wrapped `FileLineColLoc`. This is
distinct from the span-style `FusedLoc([start_loc, end_loc])` which does lose
per-location column info.

**Whether this is actually the right mechanism, and whether it is sufficient, is one
of the open questions the experiments below are intended to answer.**

The alternative — carrying scope as an op attribute rather than in the location — is
also worth considering if `FusedLoc` turns out to interact badly with location tracking
elsewhere in the pipeline.

---

## 3. Experimental Implementation Steps

### Step 1: IF scope — manual proof of concept

**Goal:** verify end-to-end that the scope marker concept produces correct LLVM IR,
`dwarfdump` output, and a working gdb session for a single `IF` statement. No recursion,
no FOR, no general mechanism — just enough to validate the design.

**3.1.1 Add the scope marker ops to the dialect**

Add `ScopeBeginOp` and `ScopeEndOp` to `silly.td`:

```tablegen
def ScopeKind : I32EnumAttr<"ScopeKind", "Scope boundary kind", [
    I32EnumAttrCase<"IfPredicate",   0>,
    I32EnumAttrCase<"IfBody",        1>,
    I32EnumAttrCase<"ElifPredicate", 2>,
    I32EnumAttrCase<"ElifBody",      3>,
    I32EnumAttrCase<"ElseBody",      4>,
    I32EnumAttrCase<"ForHeader",     5>,
    I32EnumAttrCase<"ForPredicate",  6>,
    I32EnumAttrCase<"ForBody",       7>,
]>;

def ScopeBeginOp : Silly_Op<"scope_begin"> {
    let arguments = (ins I32Attr:$id, ScopeKindAttr:$kind);
}

def ScopeEndOp : Silly_Op<"scope_end"> {
    let arguments = (ins I32Attr:$id);
}
```

Verify that a hand-written `.mlir` file containing these ops parses and round-trips
correctly through `mlir-opt` (or `silly --emit-mlir`).

**3.1.2 Hand-edit a test program's MLIR**

Take a simple silly program such as:

```
INT32 x = 3;
IF ( x < 4 )
{
    INT32 y = x + 1;
    PRINT y;
};
PRINT 42;
```

Compile it to MLIR with `silly --emit-mlir -c`. Hand-edit the resulting `.mlir` file
to insert `scope_begin`/`scope_end` markers in the expected positions, matching the
structure clang would produce:

```
// predicate scope — at the IF token location
"silly.scope_begin"() { id = 1, kind = if_predicate } : () -> ()
// ... predicate expression ops (the x < 4 comparison) ...
"silly.scope_begin"() { id = 2, kind = if_body } : () -> ()
// ... IF body ops (y declaration, PRINT y) ...
"silly.scope_end"() { id = 2 } : () -> ()
"silly.scope_end"() { id = 1 } : () -> ()
```

**3.1.3 Hack the lowering to handle this structure**

Add a minimal, non-recursive handler in the lowering pass that:

1. Before lowering a `func.func`, walks its ops looking for `ScopeBeginOp`
2. For each `ScopeBeginOp`, creates a `DILexicalBlockAttr` using the op's location and
   the enclosing `DISubprogramAttr` (or the previously created block) as parent
3. Records the mapping `id → DILexicalBlockAttr`
4. Re-stamps ops between the begin/end pair with the appropriate scope — either via
   `FusedLoc` wrapping or a separate attribute, whichever seems more natural given the
   existing lowering code
5. Erases the `ScopeBeginOp`/`ScopeEndOp` marker ops

This does not need to handle nesting, FOR loops, or any general case. It only needs to
work for the hand-edited test file.

**3.1.4 Verify**

Check the following in order, stopping if any step reveals a design problem:

- `--emit-llvm` output contains `DILexicalBlock` entries with correct line/col and
  correct parent (`DISubprogram` for the predicate block, predicate block for the body
  block)
- `dwarfdump --debug-info` shows the expected scope nesting matching the clang
  reference output for the equivalent C program
- `gdb` correctly scopes the variable `y` to the IF body — it is not visible outside
  the block, and the scope boundary appears at the right line when stepping

**Open questions to answer at this step:**

- Does `FusedLoc` wrapping preserve column info correctly through to the final
  `!DILocation` entries? Or does the translation layer need the scope in a different
  form?
- Does re-stamping the `DebugNameOp` location with the lexical block scope produce a
  correct `dbg_declare`? Or does it need to be done differently in lowering?
- Are there any ops that should *not* be re-stamped (e.g. ops that should remain at
  subprogram scope even when physically inside an IF body in the IR)?

---

### Step 2: FOR scope — second proof of concept

**Goal:** validate the same mechanism for a simple `FOR` loop. The FOR case is slightly
different from IF because it has two blocks (header and body) and because the induction
variable is currently the most visible known failure.

**3.2.1 Hand-edit a FOR test program's MLIR**

Take a program such as:

```
FOR ( INT32 i : ( 0, 5 ) )
{
    INT32 y = i + 1;
    PRINT y;
};
```

Hand-edit the MLIR to insert:

```
"silly.scope_begin"() { id = 1, kind = for_header } : () -> ()
// induction variable DebugNameOp goes here, or is moved here
"silly.scope_begin"() { id = 2, kind = for_body } : () -> ()
// ... FOR body ops ...
"silly.scope_end"() { id = 2 } : () -> ()
"silly.scope_end"() { id = 1 } : () -> ()
```

Note: the induction variable's `DebugNameOp` currently appears at an awkward location
relative to the `scf.for` structure. Part of the experiment is determining exactly where
in the op sequence it needs to appear and how the scope re-stamping interacts with it.

**3.2.2 Extend the lowering handler for FOR**

Add handling for `ForHeader` and `ForBody` kind to the minimal lowering added in step 1.
Still non-recursive.

**3.2.3 Verify**

Same verification steps as step 1, plus:

- Induction variable `i` appears in `dwarfdump` with `scope: !header_block` not
  `scope: !DISubprogram`
- gdb correctly shows `i` as a loop-scoped variable
- gdb line-stepping within the FOR body does not exhibit the ping-pong regression

---

### Step 3: General implementation (if steps 1 and 2 validate the design)

Only proceed here once the hand-edited experiments in steps 1 and 2 produce correct
results and the design questions raised in step 1 have satisfactory answers.

**3.3.1 Front-end: emit scope markers from the builder**

Add `ScopeBeginOp`/`ScopeEndOp` emission to `Builder::createIf`,
`Builder::createFor`, `Builder::finishFor`, `Builder::finishIfElifElse`, and
`Builder::enterScopedRegion`. A monotonically incrementing per-function counter
provides unique IDs. Remove the existing `DebugScopeOp` emission.

The ELIF case requires explicit care: the ELIF predicate block must be a sibling of
the IF predicate block (same parent scope), not a child. The builder knows this at
emit time and can encode it correctly in the marker sequence.

**3.3.2 Pre-lowering pass: `ScopeInstrumentationPass`**

Replace the ad-hoc lowering handler from steps 1 and 2 with a proper MLIR pass that:

- Runs before `SillyToLLVMLoweringPass`
- Walks each `func.func` maintaining a scope stack
- Handles arbitrary nesting (IF inside FOR, IF inside IF body, etc.)
- Re-stamps all ops with `FusedLoc` (or whatever mechanism the experiments validated)
- Erases all `ScopeBeginOp`/`ScopeEndOp` ops

**3.3.3 Lowering cleanup**

Update `DebugNameOp` lowering to read scope from the op's location rather than from
a `DebugScopeOp` operand. Remove `DebugScopeOp` from the dialect and all associated
builder and lowering code.

**3.3.4 Test coverage**

Add LIT tests in `tests/debug/` for:

- Single `IF` with scoped variable
- `IF`/`ELIF`/`ELSE` with variables in each branch
- Nested `IF` inside `IF` body
- `FOR` with induction variable and body variable
- `FOR` containing an `IF`
- Back-to-back `IF` statements (sibling blocks, correct parent)

---

## 4. Reference Material

For each construct under test, the reference `dwarfdump` output can be generated from
an equivalent C program:

```bash
clang -g -O0 -emit-llvm -c if.c -o if.bc
llvm-dwarfdump --debug-info if.bc
```

The key things to match in the dwarfdump output are:

- The `DW_TAG_lexical_block` entries and their `DW_AT_low_pc`/`DW_AT_high_pc` ranges
- The `DW_AT_decl_line` and scope reference for each `DW_TAG_variable`
- The parent/child nesting of lexical blocks relative to the `DW_TAG_subprogram`

### Sample clang LLVM-IR for an IF

```
#include <stdio.h>

int main(int argc, char ** argv)
{
    int x = argc;


    if ( argc )
    {
        long myScopeVar;
        myScopeVar = 1 + argc;
        printf("%ld\n", myScopeVar);
    }

    return x;
}
```

Interesting bits from the LLVM-IR:
```
  %myScopeVar = alloca i64, align 8

// !31 is a DILocation for the predicate, scoped to !32, the first DILexicalBlock.  !32 hangs off of
// !17, the DISubprogram:
  %1 = load i32, ptr %argc.addr, align 4, !dbg !31
  %tobool = icmp ne i32 %1, 0, !dbg !31
  br i1 %tobool, label %if.then, label %if.end, !dbg !31

// !36 is the DILocation for the myScopeVar declaration, scoped to the second DILexicalBlock (!34),
// which hangs off !32, the DILexicalBlock for the predicate.
if.then:                                          ; preds = %entry
    #dbg_declare(ptr %myScopeVar, !33, !DIExpression(), !36)

!7 = distinct !DICompileUnit(language: DW_LANG_C11, file: !2, producer: "clang version 22.1.1 (https://github.com/llvm/llvm-project.git fef02d48c08db859ef83f84232ed78bd9d1c323a)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !8, splitDebugInlining: false, nameTableKind: None)
!17 = distinct !DISubprogram(name: "main", scope: !2, file: !2, line: 3, type: !18, scopeLine: 4, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !7, retainedNodes: !23)
!31 = !DILocation(line: 8, column: 10, scope: !32)
!32 = distinct !DILexicalBlock(scope: !17, file: !2, line: 8, column: 10)
!33 = !DILocalVariable(name: "myScopeVar", scope: !34, file: !2, line: 10, type: !35)
!34 = distinct !DILexicalBlock(scope: !32, file: !2, line: 9, column: 5)
!36 = !DILocation(line: 10, column: 14, scope: !34)
```

* Notice that clang hoists the alloca out of the if-then region, which I don't plan to do.
* There is no hint of the use of FusedLoc here, as all the DILocation's have retained their column
  info.
* In C++ we can have variables declared in the if predicates.  Even for C, clang appears to retain
  the use of a DILexicalBlock for these predicates (where there can be no declaration in the
  predicate expression).  Is that due to a gdb dependence, or just to
  single-path the C and C++ cases?  It seems reasonable to emulate that regardless.

Full listing:
```
; ModuleID = 'if.bc'
source_filename = "if.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@.str = private unnamed_addr constant [5 x i8] c"%ld\0A\00", align 1, !dbg !0

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main(i32 noundef %argc, ptr noundef %argv) #0 !dbg !17 {
entry:
  %retval = alloca i32, align 4
  %argc.addr = alloca i32, align 4
  %argv.addr = alloca ptr, align 8
  %x = alloca i32, align 4
  %myScopeVar = alloca i64, align 8
  store i32 0, ptr %retval, align 4
  store i32 %argc, ptr %argc.addr, align 4
    #dbg_declare(ptr %argc.addr, !24, !DIExpression(), !25)
  store ptr %argv, ptr %argv.addr, align 8
    #dbg_declare(ptr %argv.addr, !26, !DIExpression(), !27)
    #dbg_declare(ptr %x, !28, !DIExpression(), !29)
  %0 = load i32, ptr %argc.addr, align 4, !dbg !30
  store i32 %0, ptr %x, align 4, !dbg !29
  %1 = load i32, ptr %argc.addr, align 4, !dbg !31
  %tobool = icmp ne i32 %1, 0, !dbg !31
  br i1 %tobool, label %if.then, label %if.end, !dbg !31

if.then:                                          ; preds = %entry
    #dbg_declare(ptr %myScopeVar, !33, !DIExpression(), !36)
  %2 = load i32, ptr %argc.addr, align 4, !dbg !37
  %add = add nsw i32 1, %2, !dbg !38
  %conv = sext i32 %add to i64, !dbg !39
  store i64 %conv, ptr %myScopeVar, align 8, !dbg !40
  %3 = load i64, ptr %myScopeVar, align 8, !dbg !41
  %call = call i32 (ptr, ...) @printf(ptr noundef @.str, i64 noundef %3), !dbg !42
  br label %if.end, !dbg !43

if.end:                                           ; preds = %if.then, %entry
  %4 = load i32, ptr %x, align 4, !dbg !44
  ret i32 %4, !dbg !45
}

declare i32 @printf(ptr noundef, ...) #1

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="non-leaf-no-reserve" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { "frame-pointer"="non-leaf-no-reserve" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }

!llvm.dbg.cu = !{!7}
!llvm.module.flags = !{!9, !10, !11, !12, !13, !14, !15}
!llvm.ident = !{!16}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(scope: null, file: !2, line: 12, type: !3, isLocal: true, isDefinition: true)
!2 = !DIFile(filename: "if.c", directory: "/home/peeter/toycalculator/tests/debug", checksumkind: CSK_MD5, checksum: "0e025876715ec489120c7663ed95e623")
!3 = !DICompositeType(tag: DW_TAG_array_type, baseType: !4, size: 40, elements: !5)
!4 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_unsigned_char)
!5 = !{!6}
!6 = !DISubrange(count: 5)
!7 = distinct !DICompileUnit(language: DW_LANG_C11, file: !2, producer: "clang version 22.1.1 (https://github.com/llvm/llvm-project.git fef02d48c08db859ef83f84232ed78bd9d1c323a)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !8, splitDebugInlining: false, nameTableKind: None)
!8 = !{!0}
!9 = !{i32 7, !"Dwarf Version", i32 5}
!10 = !{i32 2, !"Debug Info Version", i32 3}
!11 = !{i32 1, !"wchar_size", i32 4}
!12 = !{i32 8, !"PIC Level", i32 2}
!13 = !{i32 7, !"PIE Level", i32 2}
!14 = !{i32 7, !"uwtable", i32 2}
!15 = !{i32 7, !"frame-pointer", i32 4}
!16 = !{!"clang version 22.1.1 (https://github.com/llvm/llvm-project.git fef02d48c08db859ef83f84232ed78bd9d1c323a)"}
!17 = distinct !DISubprogram(name: "main", scope: !2, file: !2, line: 3, type: !18, scopeLine: 4, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !7, retainedNodes: !23)
!18 = !DISubroutineType(types: !19)
!19 = !{!20, !20, !21}
!20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !22, size: 64)
!22 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !4, size: 64)
!23 = !{}
!24 = !DILocalVariable(name: "argc", arg: 1, scope: !17, file: !2, line: 3, type: !20)
!25 = !DILocation(line: 3, column: 14, scope: !17)
!26 = !DILocalVariable(name: "argv", arg: 2, scope: !17, file: !2, line: 3, type: !21)
!27 = !DILocation(line: 3, column: 28, scope: !17)
!28 = !DILocalVariable(name: "x", scope: !17, file: !2, line: 5, type: !20)
!29 = !DILocation(line: 5, column: 9, scope: !17)
!30 = !DILocation(line: 5, column: 13, scope: !17)
!31 = !DILocation(line: 8, column: 10, scope: !32)
!32 = distinct !DILexicalBlock(scope: !17, file: !2, line: 8, column: 10)
!33 = !DILocalVariable(name: "myScopeVar", scope: !34, file: !2, line: 10, type: !35)
!34 = distinct !DILexicalBlock(scope: !32, file: !2, line: 9, column: 5)
!35 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!36 = !DILocation(line: 10, column: 14, scope: !34)
!37 = !DILocation(line: 11, column: 26, scope: !34)
!38 = !DILocation(line: 11, column: 24, scope: !34)
!39 = !DILocation(line: 11, column: 22, scope: !34)
!40 = !DILocation(line: 11, column: 20, scope: !34)
!41 = !DILocation(line: 12, column: 25, scope: !34)
!42 = !DILocation(line: 12, column: 9, scope: !34)
!43 = !DILocation(line: 13, column: 5, scope: !34)
!44 = !DILocation(line: 15, column: 12, scope: !17)
!45 = !DILocation(line: 15, column: 5, scope: !17)
```

### Sample clang LLVM-IR for a FOR

```
#include <stdio.h>

int main() // 3
{ // 4
    for ( long myLoopVar = 1 ; myLoopVar < 2 ; myLoopVar++ ) // 5
          ^                    ^
          5:11 (!26)           5:32 (!27)
               ^                         ^
               5:16 (!25)                5:42 (!29)
    { // 6
        long myScopeVar; // 7
             ^
             7:14 (!33)
        myScopeVar = 1 + myLoopVar; // 8
        printf("%ld\n", myScopeVar); // 9
    } // 10

    return 0; // 12
} // 13
```

Interesting bits from the LLVM-IR:
```
  // both variables at function level scope, no location info:
  %myLoopVar = alloca i64, align 8
  %myScopeVar = alloca i64, align 8
  // !22 is a DILocalVariable, scoped to !23
  // !23 is the first DILexicalBlock (5:5), scoped to !17 (the DISubprogram)
  // !25 is a DILocation for myLoopVar (location for the variable name, not the long)
    #dbg_declare(ptr %myLoopVar, !22, !DIExpression(), !25)


  // !26 is 5:11 (long), scoped to the first DILexicalBlock
  br label %for.cond, !dbg !26

for.cond:                                         ; preds = %for.inc, %entry
  // 5:32     [myLoopVar < 2]
  // !27 is the location, scoped to !28, the 2nd DILexicalBlock (5:5 like like !23)
  %0 = load i64, ptr %myLoopVar, align 8, !dbg !27
  // 5:42     [< 2]
  // !29 is also scoped to !28, the 2nd DILexicalBlock (block for the loop exit predicate)
  %cmp = icmp slt i64 %0, 2, !dbg !29
  // 5:5      [for]
  br i1 %cmp, label %for.body, label %for.end, !dbg !30
  //
for.body:                                         ; preds = %for.cond
    // 7:14   [myScopeVar;]
    // !31 is a DILocalVariable, scoped to !32,
    // !32 is a new DILexicalBlock (6:5) -- the third one, scoped to !28
    // !28 is the second DILexicalBlock
    #dbg_declare(ptr %myScopeVar, !31, !DIExpression(), !33)


!17 = distinct !DISubprogram(name: "main", scope: !2, file: !2, line: 3, type: !18, scopeLine: 4, spFlags: DISPFlagDefinition, unit: !7, retainedNodes: !21)
!22 = !DILocalVariable(name: "myLoopVar", scope: !23, file: !2, line: 5, type: !24)
!23 = distinct !DILexicalBlock(scope: !17, file: !2, line: 5, column: 5)
!25 = !DILocation(line: 5, column: 16, scope: !23)
!26 = !DILocation(line: 5, column: 11, scope: !23)
!27 = !DILocation(line: 5, column: 32, scope: !28)
!28 = distinct !DILexicalBlock(scope: !23, file: !2, line: 5, column: 5)
!29 = !DILocation(line: 5, column: 42, scope: !28)
!30 = !DILocation(line: 5, column: 5, scope: !23)
!31 = !DILocalVariable(name: "myScopeVar", scope: !32, file: !2, line: 7, type: !24)
!32 = distinct !DILexicalBlock(scope: !28, file: !2, line: 6, column: 5)
!33 = !DILocation(line: 7, column: 14, scope: !32)
```

* For FOR declarations, we have DILexicalBlock, the first for a FOR.  Checked that we have this,
  even for 'for ( ; predicate-expression ; post-expression )', when there is no FOR induction variable declared (or even assigned)
* We have a DILexicalBlock for any predicate expression (IF predicates or FOR exit condition).
  All the locations for that predicate are scoped to that block (but still have line:column info unlike fused locations.)
* We have a DILexicalBlock for the loop body or if/else region, used in `dbg_declare` for any variables declared.

Full LLVM-IR:
```
; ModuleID = 'for.bc'
source_filename = "for.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@.str = private unnamed_addr constant [5 x i8] c"%ld\0A\00", align 1, !dbg !0

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 !dbg !17 {
entry:
  %retval = alloca i32, align 4
  %myLoopVar = alloca i64, align 8
  %myScopeVar = alloca i64, align 8
  store i32 0, ptr %retval, align 4
    #dbg_declare(ptr %myLoopVar, !22, !DIExpression(), !25)
  store i64 1, ptr %myLoopVar, align 8, !dbg !25
  br label %for.cond, !dbg !26

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i64, ptr %myLoopVar, align 8, !dbg !27
  %cmp = icmp slt i64 %0, 2, !dbg !29
  br i1 %cmp, label %for.body, label %for.end, !dbg !30

for.body:                                         ; preds = %for.cond
    #dbg_declare(ptr %myScopeVar, !31, !DIExpression(), !33)
  %1 = load i64, ptr %myLoopVar, align 8, !dbg !34
  %add = add nsw i64 1, %1, !dbg !35
  store i64 %add, ptr %myScopeVar, align 8, !dbg !36
  %2 = load i64, ptr %myScopeVar, align 8, !dbg !37
  %call = call i32 (ptr, ...) @printf(ptr noundef @.str, i64 noundef %2), !dbg !38
  br label %for.inc, !dbg !39

for.inc:                                          ; preds = %for.body
  %3 = load i64, ptr %myLoopVar, align 8, !dbg !40
  %inc = add nsw i64 %3, 1, !dbg !40
  store i64 %inc, ptr %myLoopVar, align 8, !dbg !40
  br label %for.cond, !dbg !41, !llvm.loop !42

for.end:                                          ; preds = %for.cond
  ret i32 0, !dbg !45
}

declare i32 @printf(ptr noundef, ...) #1

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="non-leaf-no-reserve" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { "frame-pointer"="non-leaf-no-reserve" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }

!llvm.dbg.cu = !{!7}
!llvm.module.flags = !{!9, !10, !11, !12, !13, !14, !15}
!llvm.ident = !{!16}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(scope: null, file: !2, line: 9, type: !3, isLocal: true, isDefinition: true)
!2 = !DIFile(filename: "for.c", directory: "/home/peeter/toycalculator/tests/debug", checksumkind: CSK_MD5, checksum: "7fe030ad758203b8633c0184272659bb")
!3 = !DICompositeType(tag: DW_TAG_array_type, baseType: !4, size: 40, elements: !5)
!4 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_unsigned_char)
!5 = !{!6}
!6 = !DISubrange(count: 5)
!7 = distinct !DICompileUnit(language: DW_LANG_C11, file: !2, producer: "clang version 22.1.1 (https://github.com/llvm/llvm-project.git fef02d48c08db859ef83f84232ed78bd9d1c323a)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !8, splitDebugInlining: false, nameTableKind: None)
!8 = !{!0}
!9 = !{i32 7, !"Dwarf Version", i32 5}
!10 = !{i32 2, !"Debug Info Version", i32 3}
!11 = !{i32 1, !"wchar_size", i32 4}
!12 = !{i32 8, !"PIC Level", i32 2}
!13 = !{i32 7, !"PIE Level", i32 2}
!14 = !{i32 7, !"uwtable", i32 2}
!15 = !{i32 7, !"frame-pointer", i32 4}
!16 = !{!"clang version 22.1.1 (https://github.com/llvm/llvm-project.git fef02d48c08db859ef83f84232ed78bd9d1c323a)"}
!17 = distinct !DISubprogram(name: "main", scope: !2, file: !2, line: 3, type: !18, scopeLine: 4, spFlags: DISPFlagDefinition, unit: !7, retainedNodes: !21)
!18 = !DISubroutineType(types: !19)
!19 = !{!20}
!20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!21 = !{}
!22 = !DILocalVariable(name: "myLoopVar", scope: !23, file: !2, line: 5, type: !24)
!23 = distinct !DILexicalBlock(scope: !17, file: !2, line: 5, column: 5)
!24 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!25 = !DILocation(line: 5, column: 16, scope: !23)
!26 = !DILocation(line: 5, column: 11, scope: !23)
!27 = !DILocation(line: 5, column: 32, scope: !28)
!28 = distinct !DILexicalBlock(scope: !23, file: !2, line: 5, column: 5)
!29 = !DILocation(line: 5, column: 42, scope: !28)
!30 = !DILocation(line: 5, column: 5, scope: !23)
!31 = !DILocalVariable(name: "myScopeVar", scope: !32, file: !2, line: 7, type: !24)
!32 = distinct !DILexicalBlock(scope: !28, file: !2, line: 6, column: 5)
!33 = !DILocation(line: 7, column: 14, scope: !32)
!34 = !DILocation(line: 8, column: 26, scope: !32)
!35 = !DILocation(line: 8, column: 24, scope: !32)
!36 = !DILocation(line: 8, column: 20, scope: !32)
!37 = !DILocation(line: 9, column: 25, scope: !32)
!38 = !DILocation(line: 9, column: 9, scope: !32)
!39 = !DILocation(line: 10, column: 5, scope: !32)
!40 = !DILocation(line: 5, column: 57, scope: !28)
!41 = !DILocation(line: 5, column: 5, scope: !28)
!42 = distinct !{!42, !30, !43, !44}
!43 = !DILocation(line: 10, column: 5, scope: !23)
!44 = !{!"llvm.loop.mustprogress"}
!45 = !DILocation(line: 12, column: 5, scope: !17)
```

---

## Experiment Results — IF Scope Proof of Concept

### Date: March 30, 2026

### What was tested

A hand-edited MLIR file for a simple silly program:

```
INT32 x = 3;
IF ( x < 4 )
{
    INT32 myScopeVar = 1 + x;
    PRINT myScopeVar;
};
```

`ScopeBeginOp`/`ScopeEndOp` markers were inserted manually into the compiled MLIR at
the IF predicate and IF body boundaries. A minimal hack in `LoweringContext.cpp` walked
the `ScopeBeginOp` sequence after `DISubprogramAttr` creation and built two
`DILexicalBlockAttr`s — one for the IF predicate, one for the IF body — using the ops'
`FileLineColLoc` for line/column. The `DILocalVariable` for `myScopeVar` was pointed at
the body block via `FusedLoc` wrapping on the `DebugNameOp` location.

---

### Finding 1: `DILexicalBlockAttr` construction works

The two `DILexicalBlockAttr`s are constructed correctly and appear in the LLVM-IR:

```
!17 = distinct !DILexicalBlock(scope: !18, file: !1, line: 5, column: 23)
!18 = distinct !DILexicalBlock(scope: !5, file: !1, line: 3, column: 6)
```

`!18` is the IF predicate block (scoped to `!5`, the subprogram). `!17` is the IF body
block (scoped to `!18`). Parent/child nesting is correct.

---

### Finding 2: `DILocalVariable` scope via `FusedLoc` on `DebugNameOp` works

Wrapping the `DebugNameOp` location in `FusedLoc([original_loc], body_DILexicalBlockAttr)`
causes the generated `#dbg_declare` to reference the correct lexical block:

```
!16 = !DILocalVariable(name: "myScopeVar", scope: !17, ...)
!19 = !DILocation(line: 5, column: 4, scope: !17)
```

Column info is preserved through the `FusedLoc` wrapping (col 4 survives).

---

### Finding 3: All op locations must reference the lexical block scope

This was the critical discovery. When only `myScopeVar`'s `DILocalVariable` and
`#dbg_declare` location referenced `!17`, but all other op locations inside the if-body
still referenced `!5` (subprogram), LLVM's backend dropped `myScopeVar` from the DWARF
output entirely — it did not appear in `dwarfdump` and gdb could not find it.

The alloca location (entry block vs if-body block) was **not** the cause of the
failure. Moving the alloca to the entry block made no difference to the DWARF output.

When all `!DILocation` entries for ops inside the if-body were changed to reference
`!17` (the body lexical block scope), the variable survived into the final DWARF and gdb
could inspect it:

```
(gdb) p myScopeVar
$2 = 4
```

The `dwarfdump` confirmed correct output:

```
DW_TAG_lexical_block
    DW_AT_low_pc   0x00400748
    DW_AT_high_pc  <offset> 68
  DW_TAG_variable
      DW_AT_name   myScopeVar
      DW_AT_decl_line  5
      DW_AT_type   → int32_t
```

---

### Conclusion

The mechanism is validated. To produce correct DWARF lexical scopes, **every**
`!DILocation` for ops inside a scoped region must reference the innermost
`DILexicalBlock`, not the subprogram. It is not sufficient to set the scope only on the
`DILocalVariable` and its `#dbg_declare` location.

This means the re-stamping step in the `ScopeInstrumentationPass` (described in the
implementation plan) is not optional — it is required for LLVM's backend to emit
correct DWARF. The op location re-stamping must cover all ops between a
`scope_begin`/`scope_end` pair, including loads, stores, arithmetic, and print ops, not
just `DebugNameOp`.

`FusedLoc` wrapping is confirmed as the correct mechanism. It preserves column
information (the original `FileLineColLoc` is the single element inside the fused loc)
and the MLIR-to-LLVM translation layer correctly reads the `DILexicalBlockAttr` metadata
from it to populate the `scope:` field of each `!DILocation`.

---

### Remaining known issue

gdb line-stepping shows an anomaly at program start (steps to line 1 after hitting the
breakpoint at line 3). This is a pre-existing issue unrelated to lexical scope and will
be investigated separately (it may be the program start glitch observed exclusively on ARM
previously, where the ARM codegen schedules instructions from before the first breakpoint
early.)

---

## Experiment Results — IF Scope, MLIR-Driven Restamping

### Date: March 31, 2026

### What was tested

The same hand-edited MLIR IF program, but now with `ScopeBeginOp`/`ScopeEndOp` using
`I32Attr` (not `I32` SSA value) for the `id` field, and with the op-sequence restamping
loop added to the lowering hack. All ops between the if-body `scope_begin` and
`scope_end` are now wrapped in `FusedLoc([original_loc], bodyDILexicalBlockAttr)` by
iterating forward from the `ScopeBeginOp` until the matching `ScopeEndOp` is found by
`id` comparison.

The MLIR used two scope marker pairs — one for the IF predicate (id=0, kind=0) sitting
outside the `scf.if` region, and one for the IF body (id=0, kind=1) sitting inside the
`scf.if`'s then-region. The restamping loop only fires for the second pair (the body
scope), since that is where `lscope` is set.

---

### Finding 4: MLIR-driven restamping produces correct LLVM-IR and DWARF

The LLVM-IR initially showed most ops inside the if-body correctly attributed to `!15` (the body
`DILexicalBlock`):

```
  %11 = alloca i32, i64 1, align 4
    #dbg_declare(ptr %11, !18, !DIExpression(), !19)
...
!5 = distinct !DISubprogram(name: "main", linkageName: "main", scope: !1, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
...
!15 = distinct !DILexicalBlock(scope: !16, file: !1, line: 5, column: 23)
!16 = distinct !DILexicalBlock(scope: !5, file: !1, line: 3, column: 6)
!17 = !DILocation(line: 5, column: 23, scope: !15)
!18 = !DILocalVariable(name: "myScopeVar", scope: !15, file: !1, line: 5, type: !10, align: 32)
!19 = !DILocation(line: 5, column: 4, scope: !5)
!20 = !DILocation(line: 5, column: 4, scope: !15)
!21 = !DILocation(line: 7, column: 10, scope: !15)
!22 = !DILocation(line: 7, column: 4, scope: !15)
```

The single exception was the `!19` which had file-scope instead of the desired lexical scope (this
is the location for the `#dbg_declare` itself.)
The location for the dbg_declare was derived from one of the restamped locations, but was a
FileLineCol location, so lost the restamping attrs.  If that was hacked back in, using:

```
fileLoc2 = mlir::FusedLoc::get( context, { loc }, lscope );
```

(it also works to revert to just the `loc` value itself, since that has the original fusion.)

With that done, the LLVM-IR shows all ops inside the if-body correctly attributed to `!15` (the body
`DILexicalBlock`):

```
!14 = !DILocation(line: 5, column: 27, scope: !15)
!15 = distinct !DILexicalBlock(scope: !16, file: !1, line: 5, column: 23)
!16 = distinct !DILexicalBlock(scope: !5, file: !1, line: 3, column: 6)
!17 = !DILocation(line: 5, column: 23, scope: !15)
!18 = !DILocalVariable(name: "myScopeVar", scope: !15, ...)
!19 = !DILocation(line: 5, column: 4, scope: !15)
!20 = !DILocation(line: 7, column: 10, scope: !15)
!21 = !DILocation(line: 7, column: 4, scope: !15)
```

Column info is preserved through `FusedLoc` wrapping on all ops (provided the fusion is just a
single location, but scoped to the lexical block.)

The `dwarfdump` output confirms `myScopeVar` is correctly placed inside a
`DW_TAG_lexical_block`:

```
DW_TAG_lexical_block
    DW_AT_low_pc   0x00400748
    DW_AT_high_pc  <offset> 52
  DW_TAG_variable
      DW_AT_name        myScopeVar
      DW_AT_decl_line   5
      DW_AT_type        → int32_t
      DW_AT_location    (loclists, valid range 0x00400754..0x0040077c)
```

---

### Sample GDB session

```
(gdb) b main
Breakpoint 2 at 0x40073c: file ./if.silly, line 3.
(gdb) run
Starting program: /home/peeter/toycalculator/tests/prototype/if-hacked --output-directory out if-hacked.mlir  -g --emit-llvm --emit-mlir  --debug
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/lib64/libthread_db.so.1".

Breakpoint 2, main () at ./if.silly:3
3       IF ( x < 4 )
(gdb) l
1       INT32 x = 3;
2
3       IF ( x < 4 )
4       {
5          INT32 myScopeVar = 1 + x;
6
7          PRINT myScopeVar;
8       };
(gdb) b 5
Breakpoint 3 at 0x400748: file ./if.silly, line 5.
(gdb) c
Continuing.

Breakpoint 3, main () at ./if.silly:5
5          INT32 myScopeVar = 1 + x;
(gdb) n
7          PRINT myScopeVar;
(gdb) p myScopeVar
$1 = 4
(gdb) p x
$2 = 3
```

### Finding 5: Two `DILexicalBlock`s in LLVM-IR collapse to one `DW_TAG_lexical_block` in DWARF

The LLVM-IR contains two `DILexicalBlock` entries — `!16` for the IF predicate scope
and `!15` for the IF body scope. However, the final `dwarfdump` output shows only a
single `DW_TAG_lexical_block`. This matches the clang reference output for an equivalent
C program — clang's LLVM-IR also has two `DILexicalBlock`s, but only one
`DW_TAG_lexical_block` appears in the final DWARF.

LLVM's backend merges or elides the predicate-scope block when it contains no variables
and its address range is subsumed by the body-scope block. This is the expected behavior
and is not a problem — the two-block structure in the IR is still correct and necessary
to maintain proper parent/child scope relationships for the body block.

---

### Finding 6: `I32Attr` required for `id` field in scope marker ops

The tablegen definition must use `I32Attr` (a compile-time attribute) rather than `I32`
(an SSA value type) for the `id` field. Using `I32` makes `getId()` return an
`mlir::Value`, which cannot be directly compared as an integer. With `I32Attr`, `getId()`
returns `int32_t` directly, enabling the matching loop:

```cpp
if ( auto endOp = mlir::dyn_cast<silly::ScopeEndOp>( current ) )
    if ( endOp.getId() == targetId )
        break;
```

---

### Updated MLIR syntax for scope markers

```mlir
"silly.scope_begin"() <{id = 0 : i32, kind = 0 : i32}> : () -> ()  // predicate
"silly.scope_begin"() <{id = 1 : i32, kind = 1 : i32}> : () -> ()  // body
"silly.scope_end"()   <{id = 1 : i32}> : () -> ()
"silly.scope_end"()   <{id = 0 : i32}> : () -> ()
```

---

### Status

The IF scope experiment is complete and successful. The mechanism is validated:

1. `ScopeBeginOp`/`ScopeEndOp` pairs with `I32Attr` id and kind fields
2. A scope stack in the lowering handler, seeded with the `DISubprogramAttr`
3. On each `ScopeBeginOp`: create a `DILexicalBlockAttr` parented to the current stack top
4. Forward iteration from `ScopeBeginOp` to matching `ScopeEndOp`, wrapping each op's
   location in `FusedLoc([original], currentDILexicalBlockAttr)`
5. The `DebugNameOp` location wrapping causes its `DILocalVariable` to reference the
   correct scope automatically (no special-casing needed beyond the restamp loop)

---

### Next step
---

Step 2 of the implementation plan: repeat the experiment for a `FOR` loop, hand-editing
the MLIR to insert `ForHeader`/`ForPredicate`/`ForBody` scope markers. The induction
variable `DebugNameOp` placement relative to the `scf.for` structure and the three-level
block nesting (`ForHeader` → `ForPredicate` → `ForBody`) require particular attention.

## Experiment Results — FOR Scope Proof of Concept

### Date: April 1, 2026

### What was tested

A hand-edited MLIR file for a simple silly FOR loop:

```
FOR (INT64 myLoopVar : (1, 2))
{
    INT64 myScopeVar;
    myScopeVar = 1 + myLoopVar;
    PRINT myScopeVar;
};
```

`ScopeBeginOp`/`ScopeEndOp` pairs were inserted using a simplified two-scope structure
(ForHeader + ForBody only, omitting ForPredicate), since the `scf.for` bounds
computation and condition check are implicit and do not appear as explicit ops in the
MLIR at this level. The ForPredicate scope kind is reserved for future use.

---

### Scope marker placement

```
scope_begin(id=2, kind=ForHeader)    ← before scf.for, at FOR token location
%c1_i64 = ...                        ← loop start bound
%c2_i64 = ...                        ← loop end bound
%c1_i64_0 = ...                      ← loop step
scf.for %arg0 = ... {
    debug_name(%arg0) "myLoopVar"    ← inside scf.for region, before ForBody
    scope_begin(id=3, kind=ForBody)  ← at opening { location
    ...body ops...
    scope_end(id=3)
}
scope_end(id=2)                      ← after scf.for, inside func.func
```

Key decisions:
- ForHeader `scope_begin`/`scope_end` wraps the bounds computation ops and the `scf.for`
  itself, but the `scope_end` is placed after the `scf.for` closes (inside the function)
- `myLoopVar` `DebugNameOp` is placed inside the `scf.for` region before the ForBody
  `scope_begin`. This means it gets restamped with the ForHeader scope, not the ForBody
  scope, which is correct — the induction variable belongs to the loop header scope.
- ForBody `scope_begin`/`scope_end` wraps the body ops inside the `scf.for` region

---

### Results

The LLVM-IR shows correct two-level `DILexicalBlock` nesting:

```
!14 = !DILocalVariable(name: "myScopeVar", scope: !15, ...)
!15 = distinct !DILexicalBlock(scope: !16, file: !1, line: 2, column: 1)   ← ForBody
!16 = distinct !DILexicalBlock(scope: !5, file: !1, line: 1, column: 25)   ← ForHeader
```

`myLoopVar` correctly scopes to `!5` (the subprogram) in this run since the ForHeader
restamping loop does not reach into the `scf.for` region (it only iterates over sibling
ops in the same block). This is a limitation of the current hack — see note below.

The `dwarfdump` shows:

```
DW_TAG_subprogram (main)
  DW_TAG_variable
      DW_AT_name   myLoopVar          ← at subprogram scope (acceptable for now)
      DW_AT_decl_line  1
  DW_TAG_lexical_block
      DW_AT_low_pc   0x0040075c
      DW_AT_high_pc  <offset> 24
    DW_TAG_variable
        DW_AT_name   myScopeVar       ← correctly inside lexical block
        DW_AT_decl_line  3
```

The gdb session confirms both variables are accessible at the correct points:

```
(gdb) p myLoopVar
$1 = 1
(gdb) p myScopeVar
$2 = 2
```

---

### Key finding: ForHeader restamping does not cross `scf.for` region boundary

The sibling-iteration restamping loop only walks ops in the same block. The `scf.for`
op itself gets restamped (so the branch instructions lowered from it carry the ForHeader
scope), but ops inside the `scf.for` region are not reached. This means:

- The `myLoopVar` `DebugNameOp` inside the `scf.for` region is not restamped by the
  ForHeader loop — it retains the subprogram scope
- This is why `myLoopVar` appears as a `DW_TAG_variable` directly under
  `DW_TAG_subprogram` rather than inside the ForHeader lexical block

For the general implementation (Step 3), the restamping logic needs to be extended to
walk into nested regions for the ForHeader scope case. The ForBody scope restamping
already works correctly because its `scope_begin`/`scope_end` are siblings inside the
`scf.for` region, so the inner iteration loop handles them.

---

### Conclusion

The FOR scope proof of concept succeeds. Both `myLoopVar` and `myScopeVar` are
visible in gdb at the correct program points. The `DW_TAG_lexical_block` for the FOR
body is correctly nested and contains `myScopeVar`. The induction variable scoping
to the ForHeader block (rather than the subprogram) is a known remaining gap that
will be addressed in Step 3 when the restamping is extended to cross region boundaries.

The overall design — `ScopeBeginOp`/`ScopeEndOp` pairs, scope stack in lowering,
`FusedLoc` wrapping — is validated for both IF and FOR. Step 3 (general
implementation) can now proceed.

<!-- vim: set tw=100 ts=4 sw=4 et: -->
