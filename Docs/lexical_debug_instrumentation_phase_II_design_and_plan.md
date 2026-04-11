# Silly Compiler: Debug Infrastructure Implementation Plan
## Phase 2 — CF Emission, Structured Scope Regions, and Control Flow Extensions

*Document date: April 2026*
*Status: Planning — supersedes the SCF-based proof-of-concept work completed March–April 2026*

---

## 1. Background and Motivation

The Phase 1 experimentation and work, documented in the [implementation plan and experiment](lexical_scope_design_and_plan.md) validated the core `ScopeBeginOp`/`ScopeEndOp` marker approach and
established that:

- `FusedLoc([original_loc], DILexicalBlockAttr)` is the correct mechanism for
  carrying scope through to LLVM IR `!DILocation` entries
- Every op inside a scoped region must be restamped — not just the
  `DebugNameOp` — or LLVM's backend will drop the `DW_TAG_lexical_block` entirely
- The closing branch of an if-body or for-body must be tagged with a location
  scoped to the innermost `DILexicalBlock`, not the predicate block

Phase 1 left several known issues:

1. `fixBranchDebugLocs` is a post-lowering hack that works around the fact that
   `populateSCFToControlFlowConversionPatterns` owns branch emission and tags
   branches with the wrong scope
2. `break`, `continue`, and early `return` inside loop bodies cannot be
   implemented cleanly with `scf.for` / `scf.if` because SCF regions have
   structured exit semantics
3. SCF lowering does not give sufficient control of location tagging, resulting
   in dropped dwarf DI instrumentation in some cases (example: if-elif-taken.silly,
   where the line table misses line 12 completely.)
4. The `ScopeBeginOp`/`ScopeEndOp` id-matching scheme is a maintenance burden
   and does not naturally express the parent/child relationship between scopes
5. The Bison frontend is broken and needs to be brought to parity with the
   ANTLR4 frontend once the architecture stabilises
6. The `FusedLoc` approach for function boundaries (used for `DISubprogramAttr`
   attachment) and for lexical blocks should be unified and documented

This document describes the next phase of work, structured as a sequence of
small, verifiable steps in the same exploratory style as Phase 1.

---

## 2. Architectural Direction

### 2.1 Replace SCF with direct CF emission

The `scf.for` and `scf.if` ops will be replaced with direct emission of
`cf.br` and `cf.cond_br` ops. This gives full control over:

- Which location and scope is attached to every branch
- The block structure (enabling `break`, `continue`, early `return`)
- The placement of `ScopeBeginOp`/`ScopeEndOp` relative to the branch targets

The SCF conversion patterns (`populateSCFToControlFlowConversionPatterns`,
`populateControlFlowToLLVMConversionPatterns`) will be removed from the
lowering pass.

### 2.2 Replace id-matched ScopeBegin/ScopeEnd with region-bearing scope op

Rather than matching `ScopeBeginOp`/`ScopeEndOp` pairs by integer id, introduce
a single `ScopeOp` that carries its body as an MLIR region:

```tablegen
def Silly_ScopeOp : Op<Silly_Dialect, "scope", [SingleBlock]> {
    let summary = "Lexical scope region for debug instrumentation.";
    let arguments = (ins);
    let regions = (region SizedRegion<1>:$body);
}
```

The scope's begin location is the op's own location (set to the `{` token).
The scope's end location is carried as a second location on the op — using the
same `FusedLoc([begin_loc, end_loc])` pattern already used for `DISubprogramAttr`
attachment on `func.func` ops.

This eliminates the id-matching loop entirely. The pre-lowering walk simply
recurses into `ScopeOp` regions, and the `ScopeOp` is erased (inlined) after
restamping.

### 2.3 Unify FusedLoc usage for scope metadata

The pattern used for `DISubprogramAttr` attachment:

```cpp
funcOp->setLoc(builder.getFusedLoc({beginLoc, endLoc}, subAttr));
```

will be mirrored for `DILexicalBlockAttr` attachment on `ScopeOp`:

```cpp
scopeOp->setLoc(builder.getFusedLoc({openBraceLoc, closeBraceLoc}, {}));
```

The pre-lowering walk reads both locations from the `FusedLoc` on the `ScopeOp`
and uses `closeBraceLoc` as the location for the terminating branch of the
scope's lowered basic block — replacing `fixBranchDebugLocs` entirely.

### 2.4 Control flow block structure

For each control flow construct, the CF emission will produce a fixed block
structure that makes `break`, `continue`, and early `return` straightforward:

**IF/ELIF/ELSE:**
```
^pred_block:          // predicate computation, scope = predicate DILexicalBlock
    cf.cond_br %cond, ^then_block, ^else_block

^then_block:          // body, scope = body DILexicalBlock
    ...
    cf.br ^merge_block   // location = closing }, scope = body DILexicalBlock

^else_block:          // elif predicate or else body
    ...
    cf.br ^merge_block

^merge_block:         // post-if, scope = enclosing scope
    ...
```

**FOR:**
```
^for_header:          // induction var init, scope = header DILexicalBlock
    cf.br ^for_cond

^for_cond:            // exit predicate, scope = predicate DILexicalBlock
    cf.cond_br %cond, ^for_body, ^for_end

^for_body:            // body, scope = body DILexicalBlock
    ...
    cf.br ^for_inc    // location = closing }, scope = body DILexicalBlock

^for_inc:             // post-increment, scope = header DILexicalBlock
    cf.br ^for_cond

^for_end:             // post-loop, scope = enclosing scope
    ...
```

`break` lowers to `cf.br ^for_end`.
`continue` lowers to `cf.br ^for_inc`.
Early `return` lowers to `cf.br ^exit_block` (see section 5).

---

## 3. Step-by-Step Implementation Plan

### Step 1: Introduce `ScopeOp` with region body

**Goal:** replace the id-matched `ScopeBeginOp`/`ScopeEndOp` pair with a
single region-bearing `ScopeOp`. No CF changes yet — this step works entirely
within the existing SCF-based lowering.

**3.1.1 Add `ScopeOp` to the dialect**

In `silly.td`, add:

```tablegen
def Silly_ScopeOp : Op<Silly_Dialect, "scope", [SingleBlock]> {
    let summary = "Lexical scope region.";
    let description = [{
        Wraps a sequence of ops in a lexical scope. The op's location is a
        FusedLoc([open_brace_loc, close_brace_loc]) carrying the source
        positions of the opening and closing braces.  The pre-lowering
        ScopeInstrumentationPass uses these to construct a DILexicalBlockAttr
        and restamp all ops in the region with the appropriate FusedLoc.
    }];
    let regions = (region SizedRegion<1>:$body);
}
```

Keep `ScopeBeginOp` and `ScopeEndOp` in the dialect temporarily for the Bison
frontend.

**3.1.2 Update the ANTLR4 frontend to emit `ScopeOp`**

In `Builder::createIf`, `Builder::createFor`, `Builder::selectElseBlock`:

- Replace the `ScopeBeginOp::create` + `ScopeEndOp::create` + manual IP
  manipulation with `ScopeOp::create`
- Set the `ScopeOp` location to `builder.getFusedLoc({openBraceLoc, closeBraceLoc})`
- Move the body ops into the `ScopeOp`'s region by setting the insertion point
  to the region's entry block

**3.1.3 Update `processScopeBegin` to handle `ScopeOp`**

The pre-lowering walk in `LoweringContext::processScopedOps` currently matches
`ScopeBeginOp` and searches forward for the matching `ScopeEndOp`. Replace this
with a direct region walk:

```cpp
if (auto scopeOp = mlir::dyn_cast<silly::ScopeOp>(op)) {
    auto fusedLoc = mlir::cast<mlir::FusedLoc>(scopeOp.getLoc());
    // fusedLoc.getLocations()[0] = open brace loc
    // fusedLoc.getLocations()[1] = close brace loc
    mlir::Location openLoc  = fusedLoc.getLocations()[0];
    mlir::Location closeLoc = fusedLoc.getLocations()[1];

    auto flc = mlir::cast<mlir::FileLineColLoc>(openLoc);
    auto thisScope = mlir::LLVM::DILexicalBlockAttr::get(
        context, parentScope, fileAttr, flc.getLine(), flc.getColumn());

    mlir::Location closingFusedLoc =
        mlir::FusedLoc::get(context, {closeLoc}, thisScope);
    blockClosingLoc[&scopeOp.getBody().front()] = closingFusedLoc;

    processScopedOps(scopeOp.getBody().front().begin(),
                     scopeOp.getBody().front().end(),
                     thisScope);
}
```

The `blockClosingLoc` map entry is now keyed on the `ScopeOp`'s region block,
and `fixBranchDebugLocs` stamps the terminator of that block after SCF→CF
lowering inlines the region.

**3.1.4 Add `ScopeOpLowering` — inline the region**

The `ScopeOp` needs to be lowered by inlining its region body into the parent
block:

```cpp
class ScopeOpLowering : public mlir::ConversionPattern {
    mlir::LogicalResult matchAndRewrite(
        mlir::Operation* op,
        mlir::ArrayRef<mlir::Value> operands,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
        auto scopeOp = mlir::cast<silly::ScopeOp>(op);
        rewriter.inlineBlockBefore(
            &scopeOp.getBody().front(), op, {});
        rewriter.eraseOp(op);
        return mlir::success();
    }
};
```

**3.1.5 Verify**

- MLIR output should show `silly.scope { ... }` blocks in the correct positions
- LLVM-IR and dwarfdump output should be identical to the current `ScopeBeginOp`
  / `ScopeEndOp` output
- `fixBranchDebugLocs` should still be required at this step (it will be removed
  in Step 3)

---

### Step 2: Switch IF emission to direct CF

**Goal:** replace `scf.if` emission with direct `cf.cond_br` emission for `IF`
statements. FOR loops remain on SCF at this step.

**3.2.1 Add block creation helpers to `Builder`**

```cpp
mlir::Block* Builder::createBlock(mlir::Region* region, mlir::Location loc);
```

This creates a new block at the end of the given region and returns it. All IF
and FOR block creation will use this helper.

**3.2.2 Rewrite `Builder::createIf`**

Replace:
```cpp
mlir::scf::IfOp ifOp = mlir::scf::IfOp::create(...);
```

With direct CF emission:

```cpp
void Builder::createIf(mlir::Location predLoc, mlir::Location openBraceLoc,
                        mlir::Location closeBraceLoc,
                        mlir::Value conditionPredicate,
                        mlir::Operation* retOp, LocationStack& ls)
{
    mlir::func::FuncOp funcOp = getCurrentFuncOp();
    mlir::Region& funcRegion = funcOp.getBody();

    mlir::Block* thenBlock  = createBlock(&funcRegion, openBraceLoc);
    mlir::Block* elseBlock  = createBlock(&funcRegion, openBraceLoc);
    mlir::Block* mergeBlock = createBlock(&funcRegion, closeBraceLoc);

    // Emit the conditional branch in the current block
    mlir::cf::CondBranchOp::create(builder, predLoc,
        conditionPredicate, thenBlock, elseBlock);

    // Push merge block as the insertion point to restore after IF body
    f.pushToInsertionPointStack(retOp);
    f.recordMergeBlock(mergeBlock);

    // Emit ScopeOp for the then-body and set IP inside it
    builder.setInsertionPointToStart(thenBlock);
    emitScopeOp(openBraceLoc, closeBraceLoc);
}
```

**3.2.3 Rewrite `Builder::selectElseBlock`**

```cpp
void Builder::selectElseBlock(mlir::Location loc,
                               mlir::Location openBraceLoc,
                               mlir::Location closeBraceLoc)
{
    // The current block is the then-body.  Emit its closing branch.
    mlir::Block* mergeBlock = f.currentMergeBlock();
    mlir::cf::BranchOp::create(builder, closeBraceLoc, mergeBlock);

    // Set IP to the else block
    mlir::Block* elseBlock = f.currentElseBlock();
    builder.setInsertionPointToStart(elseBlock);
    emitScopeOp(openBraceLoc, closeBraceLoc);
}
```

**3.2.4 Rewrite `Builder::finishIfElifElse`**

```cpp
void Builder::finishIfElifElse(mlir::Location loc)
{
    // Emit the closing branch for the last else/elif body
    mlir::Block* mergeBlock = f.currentMergeBlock();
    mlir::cf::BranchOp::create(builder, loc, mergeBlock);

    // Restore IP to after the merge block
    builder.setInsertionPointToStart(mergeBlock);
    f.popFromInsertionPointStack(builder);
}
```

**3.2.5 Remove SCF from the IF lowering path**

`scf.if` is no longer emitted, so `populateSCFToControlFlowConversionPatterns`
can be removed from the conversion target (or kept temporarily for FOR until
Step 3). The `ScopeOpLowering` handles the `silly.scope` region inline.

**3.2.6 Remove `fixBranchDebugLocs` for IF**

With direct CF emission, the closing branch of the then-body is emitted by
`selectElseBlock` with `closeBraceLoc` as its location and the body
`DILexicalBlock` as its scope (from the `ScopeOp` restamping). The hack is no
longer needed for IF.

Verify that the dwarfdump output for `if-with-decl.silly` and
`if-elif-taken.silly` is correct without `fixBranchDebugLocs`.

---

### Step 3: Switch FOR emission to direct CF

**Goal:** replace `scf.for` with direct CF block emission. This is the step
that enables `break` and `continue`.

**3.3.1 Rewrite `Builder::createFor`**

```cpp
void Builder::createFor(mlir::Location loc, mlir::Location openBraceLoc,
                         mlir::Location closeBraceLoc,
                         const std::string& varName, mlir::Type elemType,
                         mlir::Location varLoc, mlir::Operation* retOp,
                         mlir::Value start, mlir::Value end, mlir::Value step,
                         LocationStack& ls)
{
    mlir::func::FuncOp funcOp = getCurrentFuncOp();
    mlir::Region& funcRegion = funcOp.getBody();

    mlir::Block* headerBlock = createBlock(&funcRegion, loc);
    mlir::Block* condBlock   = createBlock(&funcRegion, loc);
    mlir::Block* bodyBlock   = createBlock(&funcRegion, openBraceLoc);
    mlir::Block* incBlock    = createBlock(&funcRegion, closeBraceLoc);
    mlir::Block* endBlock    = createBlock(&funcRegion, closeBraceLoc);

    // headerBlock: initialise induction variable, jump to condBlock
    builder.setInsertionPointToStart(headerBlock);
    mlir::LLVM::AllocaOp inductionAlloca = createAlloca(...);
    mlir::LLVM::StoreOp::create(builder, varLoc, start, inductionAlloca);
    silly::DebugNameOp::create(builder, varLoc, inductionAlloca, varName);
    // ScopeOp for the header (induction var scope)
    emitScopeOp(loc, closeBraceLoc);
    cf::BranchOp::create(builder, loc, condBlock);

    // condBlock: load induction var, compare, branch
    builder.setInsertionPointToStart(condBlock);
    mlir::Value iv = mlir::LLVM::LoadOp::create(builder, loc, elemType, inductionAlloca);
    mlir::Value cond = createBinaryCompare(loc, CmpBinOpKind::Less, iv, end, ls);
    cf::CondBranchOp::create(builder, loc, cond, bodyBlock, endBlock);

    // bodyBlock: push break/continue targets, emit ScopeOp
    builder.setInsertionPointToStart(bodyBlock);
    f.pushLoopBlocks(endBlock, incBlock);  // for break/continue
    emitScopeOp(openBraceLoc, closeBraceLoc);

    // incBlock and endBlock will be wired up in finishFor
    f.pushToInsertionPointStack(retOp);
    f.recordLoopIncBlock(incBlock);
    f.recordLoopEndBlock(endBlock);
    f.recordLoopInductionAlloca(varName, inductionAlloca);
}
```

**3.3.2 Rewrite `Builder::finishFor`**

```cpp
void Builder::finishFor(mlir::Location closeLoc)
{
    // Close the body block
    mlir::Block* incBlock = f.currentLoopIncBlock();
    cf::BranchOp::create(builder, closeLoc, incBlock);

    // incBlock: increment induction var, jump back to cond
    mlir::Block* condBlock = f.currentLoopCondBlock();
    builder.setInsertionPointToStart(incBlock);
    // load, add step, store
    cf::BranchOp::create(builder, closeLoc, condBlock);

    // Restore IP to after the loop
    mlir::Block* endBlock = f.currentLoopEndBlock();
    builder.setInsertionPointToStart(endBlock);
    f.popFromInsertionPointStack(builder);
    f.popLoopBlocks();
    f.popInductionVariable();
}
```

**3.3.3 Remove remaining SCF from the lowering pass**

At this point `scf.for` and `scf.if` are no longer emitted. Remove
`populateSCFToControlFlowConversionPatterns` from the lowering pass entirely.
The `mlir::scf::SCFDialect` can be removed from the registered dialects.

**3.3.4 Remove `fixBranchDebugLocs` entirely**

With direct CF emission, every branch is emitted with the correct location and
scope at construction time. The hack can be deleted.

**3.3.5 Verify FOR scope**

Check that:

- Induction variable `DILocalVariable` references the ForHeader `DILexicalBlock`
- Body variable `DILocalVariable` references the ForBody `DILexicalBlock`
- `dwarfdump` shows correct nested `DW_TAG_lexical_block` entries
- gdb correctly scopes `i` to the loop and `y` to the body

---

### Step 4: Add `BREAK` and `CONTINUE`

**Goal:** implement `BREAK` and `CONTINUE` statements using the loop block
references established in Step 3.

**3.4.1 Add `BREAK` to the grammar and frontend**

In the ANTLR4 grammar, add `BREAK_TOKEN` and a `breakStatement` rule. In the
frontend:

```cpp
void Builder::createBreak(mlir::Location loc, LocationStack& ls)
{
    mlir::Block* endBlock = f.currentLoopEndBlock();
    if (!endBlock) {
        emitUserError(loc, "BREAK outside of FOR loop", currentFuncName);
        return;
    }
    cf::BranchOp::create(builder, loc, endBlock);

    // Create an unreachable block to receive subsequent ops
    // (prevents builder from appending to the terminated block)
    mlir::Block* deadBlock = createBlock(&getCurrentFuncOp().getBody(), loc);
    builder.setInsertionPointToStart(deadBlock);
}
```

**3.4.2 Add `CONTINUE` to the grammar and frontend**

Identical structure to `BREAK`, but branches to `incBlock` instead of
`endBlock`:

```cpp
void Builder::createContinue(mlir::Location loc, LocationStack& ls)
{
    mlir::Block* incBlock = f.currentLoopIncBlock();
    cf::BranchOp::create(builder, loc, incBlock);
    mlir::Block* deadBlock = createBlock(&getCurrentFuncOp().getBody(), loc);
    builder.setInsertionPointToStart(deadBlock);
}
```

**3.4.3 Dead block cleanup pass**

After lowering, blocks with no predecessors and no ops other than an implicit
terminator should be eliminated. Add a simple dead block removal pass:

```cpp
void removeDeadBlocks(mlir::func::FuncOp funcOp) {
    funcOp.walk([](mlir::Block* block) {
        if (block->hasNoPredecessors() && !block->isEntryBlock())
            block->erase();
    });
}
```

**3.4.4 Verify**

Test cases:

- `FOR` with `BREAK` in the middle — loop exits, post-loop code runs
- `FOR` with `CONTINUE` — loop iteration skips remainder of body
- Nested `FOR` with `BREAK` — inner break does not affect outer loop
- gdb stepping through a loop with `BREAK`

---

### Step 5: Early function return from nested scope

**Goal:** allow `RETURN` statements inside `IF` bodies and `FOR` bodies to
correctly exit the function. This requires a landing block at the function exit
that all return paths branch to.

**3.5.1 Add a function exit block**

In `Builder::createNewFunctionState`, create a dedicated exit block:

```cpp
mlir::Block* exitBlock = createBlock(&funcOp.getBody(), funcEndLoc);
f.setExitBlock(exitBlock);
```

All `RETURN` statements branch to this block, passing the return value as a
block argument:

```cpp
// exitBlock has one block argument: the return value
exitBlock->addArgument(returnType, funcEndLoc);
```

**3.5.2 Rewrite `Builder::createReturn`**

```cpp
void Builder::createReturn(mlir::Location loc, mlir::Value retVal,
                            LocationStack& ls)
{
    mlir::Block* exitBlock = f.getExitBlock();
    cf::BranchOp::create(builder, loc, exitBlock, mlir::ValueRange{retVal});

    // Dead block for subsequent ops
    mlir::Block* deadBlock = createBlock(&getCurrentFuncOp().getBody(), loc);
    builder.setInsertionPointToStart(deadBlock);
}
```

**3.5.3 Emit the actual `func.return` in the exit block**

At the end of `Builder::finishFunction`, wire up the exit block:

```cpp
builder.setInsertionPointToStart(f.getExitBlock());
mlir::Value retVal = f.getExitBlock()->getArgument(0);
mlir::func::ReturnOp::create(builder, funcEndLoc, retVal);
```

**3.5.4 Verify**

Test cases:

- `RETURN` inside an `IF` body
- `RETURN` inside a `FOR` body
- `RETURN` inside nested `IF` inside `FOR`
- Multiple `RETURN` paths with different values
- gdb stepping to a `RETURN` inside an `IF`

---

### Step 6: Expand test coverage

**3.6.1 Convert prototype hacked files to LIT tests**

For each program in `prototypes/lexicalblock/`, create a corresponding LIT test
in `tests/debug/` that:

- Compiles the `.silly` file with `-g`
- Runs `llvm-dwarfdump --debug-info` and checks for expected `DW_TAG_lexical_block`
  entries and `DW_TAG_variable` scope references
- Optionally runs a gdb expect script for interactive stepping verification

**3.6.2 Minimum test matrix**

| Test file | Checks |
|-----------|--------|
| `if-with-decl.silly` | `y` scoped to body lexical block, `x` at subprogram |
| `if-elif-taken.silly` | All three scopes present, line table has line 12 |
| `for-with-decl.silly` | Induction var in header block, body var in body block |
| `nested-if-in-for.silly` | Three-level nesting, all vars visible in gdb |
| `break-in-for.silly` | Loop exits correctly, post-loop code reachable |
| `continue-in-for.silly` | Iteration skips body remainder correctly |
| `early-return-in-if.silly` | Return from inside IF exits function |
| `nested-scopes-same-name.silly` | Variable shadowing — correct gdb `p` result |

**3.6.3 Logical operator location attribution**

As noted in the TODO list, chains of `&&`, `||`, `^`, `&` do not yet use token
locations for each operator. Add token location tracking for
`BOOLEANOR_TOKEN`, `BOOLANAND_TOKEN`, `BOOLEANXOR_TOKEN` in the same style as
the binary arithmetic and comparison improvements from Phase 1.

---

### Step 7: Fix the Bison frontend (deferred)

The Bison frontend is deliberately left broken after Phase 1 and will remain so
through Phase 2. The intent is to bring it to parity with the ANTLR4 frontend
in a single pass once the ANTLR4 architecture has stabilised after Step 3.

The Bison FE changes required will mirror the ANTLR4 changes:

- Replace `DebugScopeOp` usage with `ScopeOp` region emission
- Pass brace token locations to `createIf`/`createFor`/`selectElseBlock`
- Replace `enterScopedRegion`/`exitScopedRegion` with
  `createNewVariableLookupScope`/`removeCurrentVariableLookupScope`
- Add `BREAK`, `CONTINUE` grammar rules
- Add early return exit block wiring in `finishFunction`

This is tracked separately and will be scheduled after Step 5 is complete.

---

## 4. Data Structure Changes

### 4.1 `ParserPerFunctionState` additions

```cpp
// Loop block targets for break/continue
struct LoopBlocks {
    mlir::Block* condBlock;
    mlir::Block* incBlock;
    mlir::Block* endBlock;
};
std::vector<LoopBlocks> loopBlockStack;

// Merge blocks for IF/ELIF/ELSE
std::vector<mlir::Block*> mergeBlockStack;

// Function exit block for early return
mlir::Block* exitBlock{};
```

### 4.2 `LoweringContext` changes

- Remove `blockClosingLoc` map (no longer needed after Step 3)
- Remove `fixBranchDebugLocs` (no longer needed after Step 2)
- `processScopedOps` updated to handle `ScopeOp` regions directly
- `ScopeBeginOp`/`ScopeEndOp` processing retained temporarily for Bison FE
  compatibility until Step 7

### 4.3 `Builder` additions

```cpp
// Block creation helper
mlir::Block* createBlock(mlir::Region* region, mlir::Location loc);

// Scope op emission helper
void emitScopeOp(mlir::Location openLoc, mlir::Location closeLoc);

// Loop context accessors
mlir::Block* currentLoopCondBlock();
mlir::Block* currentLoopIncBlock();
mlir::Block* currentLoopEndBlock();
void pushLoopBlocks(mlir::Block* end, mlir::Block* inc);
void popLoopBlocks();
```

---

## 5. Open Questions

**Q1: `ScopeOp` region vs. flat op sequence**

The region-bearing `ScopeOp` is cleaner than id-matched pairs but requires
that all body ops be emitted inside the region. This means the builder must
manage insertion points carefully when `ScopeOp`s nest. Is this feasible given
the current builder structure, or does it require a more significant refactor
of how the builder tracks its current insertion point?

*Proposed answer:* The existing `insertionPointStack` mechanism already handles
this — the `ScopeOp` region's entry block becomes the insertion point, and
`finishScope` pops back to the parent block. The region inlining in
`ScopeOpLowering` happens before SCF/CF lowering and restores the flat block
structure.

**Q2: Dead block elimination and debug info**

Dead blocks (created after `BREAK`/`CONTINUE`/`RETURN`) will have no location
or scope information. Should they be removed before or after the
`ScopeInstrumentation` walk?

*Proposed answer:* Remove them after the scope walk but before the conversion
pass. A dead block with no ops and no predecessors can be erased without
affecting any debug metadata.

**Q3: FOR with non-unit step and induction variable type**

The current implementation casts start/end/step to the induction variable type.
With CF emission, the load/store approach for the induction variable (rather
than an SSA block argument) needs to handle the same casts. Ensure the cast
logic is preserved in the CF rewrite.

**Q4: ELIF predicate scope nesting**

In the current ANTLR4 frontend, ELIF creates a nested `scf.if` inside the else
region of the outer `scf.if`. With direct CF emission, the ELIF predicate block
is a sibling of the IF then-block, not a child. The `DILexicalBlock` parent for
the ELIF predicate should be the enclosing function or outer IF predicate block,
not the outer IF body block. Verify this is handled correctly in the CF
emission.

---

## 6. Reference: Scope Block Nesting Expected by LLVM Backend

For reference, the expected `DILexicalBlock` nesting for each construct,
derived from clang reference output:

### IF/ELIF/ELSE
```
DISubprogram (function)
  DILexicalBlock (IF predicate — at IF token)
    DILexicalBlock (IF body — at opening {)
  DILexicalBlock (ELIF predicate — at ELIF token, parent = subprogram)
    DILexicalBlock (ELIF body — at opening {)
  DILexicalBlock (ELSE body — at opening {, parent = subprogram)
```

### FOR
```
DISubprogram (function)
  DILexicalBlock (FOR header — at FOR token, contains induction var)
    DILexicalBlock (FOR predicate — at FOR token same position)
      DILexicalBlock (FOR body — at opening {)
```

All op locations inside a scope must reference the innermost enclosing
`DILexicalBlock`. The closing branch of each body block must reference that
body's `DILexicalBlock` with the closing `}` source location.

---

## 7. Execution Order Summary

| Step | Description | Removes hack? | Enables feature? |
|------|-------------|---------------|-----------------|
| 1 | `ScopeOp` region-bearing op | Simplifies id matching | — |
| 2 | IF → direct CF emission | Removes `fixBranchDebugLocs` for IF | — |
| 3 | FOR → direct CF emission | Removes `fixBranchDebugLocs` entirely, removes SCF | — |
| 4 | `BREAK` / `CONTINUE` | — | BREAK, CONTINUE |
| 5 | Early function return | — | RETURN from nested scope |
| 6 | Test expansion + logical op locations | — | Better debug quality |
| 7 | Bison FE parity | — | Bison FE working again |

Each step is independently verifiable. Steps 1–3 are purely refactoring with
no user-visible behaviour change. Steps 4–5 add new language features. Step 6
hardens the implementation. Step 7 restores a broken component.

<!-- vim: set tw=100 ts=4 sw=4 et: -->
