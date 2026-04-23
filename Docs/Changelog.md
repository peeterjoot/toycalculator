# Changelog V0.11.0 (WIP)

## Bug fix:

```
INT64 y = 3;
BOOL lt2 = (y < 10);
```

Had inappropriate i1 narrowing for the literal creation in bowls of parseExpression (both front ends.)

## IF/ELIF/ELSE/FOR converted from SCF dialect to CF.

* This provides infrastructure to allow for early return/exit, and FOR break/continue.  It also gives more control for FOR step direction, and will allow for an easy implementation of DO or WHILE loops.

### FOR step direction.

* Negative step sizes are now supported.  A simple constant folding for step expressions was implemented, supporting
  only negation.  If that folding "fails" to match expectations (i.e.: constant or negated constant) an error is raised.
  Zero step size is now checked for, so such a program will no longer infinite loop.

## Bison FE.

* Fixed all the non-debug/syntax-error testsuite failures.

## Debugging.

* Updated the dwarf types to use the silly-language type names.

## Lexical scope DWARF instrumentation: ScopeBegin/ScopeEnd markers and restamping

Replace the previous DebugScopeOp approach (which threaded scope information
through the SSA value stream as an operand to DebugNameOp) with a cleaner
ScopeBeginOp/ScopeEndOp marker pair approach.  The new mechanism is validated
by the IF and FOR proof-of-concept experiments documented in the implementation
plan.

### Mechanism

ScopeBeginOp and ScopeEndOp are emitted by the front end to mark lexical scope
boundaries in the op sequence.  A pre-lowering walk in LoweringContext traverses
the op sequence, maintains a DIScope stack seeded with the DISubprogramAttr, and
for each ScopeBeginOp:

  - Creates a DILexicalBlockAttr parented to the current stack top
  - Restamps every op between the begin/end pair with a FusedLoc wrapping the
    original FileLineColLoc and the new DILexicalBlockAttr
  - Records any DebugNameOp found in the range in a DenseMap keyed by
    Operation\*, for use by constructVariableDI
  - Recurses into nested regions (scf.for, scf.if) so that ForHeader scope
    correctly reaches the induction variable DebugNameOp
  - Records the ScopeEndOp location (fused with the innermost DILexicalBlockAttr)
    in blockClosingLoc, keyed by the enclosing Block*

ScopeBeginOp and ScopeEndOp are erased later via LowerByDeletion patterns in
the conversion pass, keeping the pre-lowering walk non-mutating (other than
location restamping).

### Front end changes (ANTLR4 FE only; Bison FE known broken)

- ScopeBeginOp/ScopeEndOp pairs emitted from createIf, createFor, selectElseBlock
- Brace token locations (`LEFT_CURLY_BRACKET_TOKEN`, `RIGHT_CURLY_BRACKET_TOKEN`)
  passed to createIf/createFor/selectElseBlock for correct DILexicalBlock
  positioning (predicate block at IF token, body block at opening brace)
- DebugScopeOp and its associated SSA scope operand on DebugNameOp removed
- enterScopedRegion/exitScopedRegion replaced with
  createNewVariableLookupScope/removeCurrentVariableLookupScope (variable
  lookup scoping decoupled from debug scope marker emission)
- Binary operator and comparison token locations now used for icmp/arith op
  locations, matching clang's LLVM-IR column attribution

### Alloca hoisting

DeclareOpLowering now calls LoweringContext::createAlloca which inserts all
allocas at the start of the function entry block (after any previously inserted
allocas), matching clang's -O0 behavior.  This eliminates the alloca-in-if-body
pattern that was a latent risk for some LLVM backend versions.

### Known issues / follow-on work

- Bison FE does not yet emit ScopeBeginOp/ScopeEndOp; it is broken by this
  change and will be fixed in a follow-on commit
- The SCF-to-CF conversion still owns branch emission, requiring the
  fixBranchDebugLocs hack.  A planned transition from SCF to direct CF emission
  will eliminate this hack and also enable break/continue in FOR loops
- FOR induction variable scoping to the ForHeader DILexicalBlock (rather than
  the DISubprogram) requires the region-crossing restamping path; this is
  implemented but needs a dedicated test
