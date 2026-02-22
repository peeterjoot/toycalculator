# Silly Language: IMPORT Module System — Design Notes and Ideas

## Overview

The proposed feature adds an `IMPORT` statement to the Silly language:

```
IMPORT foo;
```

The mechanism:

- Every module compiles to a `.sir` (or `.mlirbc`) as an intermediate representation — analogous to Java bytecode or a `.hi` Haskell interface file.
- When the compiler encounters `IMPORT foo`, it locates `foo.sir`, builds it if required (with timestamp checking), then walks the `mlir::Module` for all `FuncOp`s and clones the prototype (declaration-only) form of each into the importing module.
- Cycle detection is required, with a visited map.

---

## Phase 0: Recommended First Step — `.mlirbc` Export/Import in the Driver

Before tackling the full IMPORT system, add `.mlirbc` (MLIR bytecode) read/write support to the driver as a standalone feature. This:

- Validates the round-trip serialization story before the module system depends on it.
- Gives you a fast, compact intermediate format from day one.
- Is low-risk and largely mechanical.

MLIR has built-in bytecode support:

```cpp
// Write
mlir::writeBytecodeToFile( module, outputStream );

// Read — handles both text .mlir and binary .mlirbc automatically
mlir::parseSourceFile<mlir::ModuleOp>( sourceMgr, context );
```

The `.sir` extension can map to either the textual or bytecode format depending on a driver flag, or you could use `.sir` for text and `.sirbc` for binary. Settle this before the module system is built on top of it.

---

## Cloning a FuncOp as a Prototype

The natural instinct is to use `FuncOp::clone()`, but that copies the region (body) too. Instead, construct a new `FuncOp` with the same name and type and leave the body empty. A newly created `FuncOp` with no `addEntryBlock()` call is already a prototype — `isExternal()` returns `true` whenever the region is empty.

```cpp
void importPrototype( mlir::func::FuncOp srcFuncOp,
                      mlir::ModuleOp    dstModule,
                      mlir::OpBuilder&  builder )
{
    // Skip private functions — not part of the module's public API
    if ( srcFuncOp.isPrivate() )
        return;

    // Skip if already imported (idempotent)
    if ( dstModule.lookupSymbol( srcFuncOp.getSymName() ) )
        return;

    mlir::OpBuilder::InsertionGuard guard( builder );
    builder.setInsertionPointToStart( dstModule.getBody() );

    auto protoOp = builder.create<mlir::func::FuncOp>(
        srcFuncOp.getLoc(),
        srcFuncOp.getSymName(),
        srcFuncOp.getFunctionType()
    );

    protoOp.setPrivate(); // marks as external declaration
}

void importAllPrototypes( mlir::ModuleOp   srcModule,
                          mlir::ModuleOp   dstModule,
                          mlir::OpBuilder& builder )
{
    srcModule.walk( [&]( mlir::func::FuncOp funcOp )
    {
        if ( !funcOp.isExternal() ) // only import defined functions, not re-exported decls
            importPrototype( funcOp, dstModule, builder );
    } );
}
```

---

## Design Concerns and Notes

### 1. Bytecode Format — Pin Early

MLIR's textual `.mlir` format is human-readable but not designed for fast parsing at scale. The binary bytecode format (`.mlirbc`) is built into MLIR and should be preferred for `.sir` files once the system matures. See Phase 0 above. Switching formats later is straightforward, but the sooner the driver supports both read and write, the sooner everything else can rely on it.

### 2. Visibility Semantics — Decide on an Export Model

Right now `private` maps naturally to "not exported." You need a clear rule for what gets exported from a module:

- **Implicit export**: everything non-private is public (Java model). Simple, but internal helpers leak into the public API.
- **Explicit export**: require an `EXPORT` keyword or annotation on definitions intended to be importable (Haskell/Rust model). More disciplined, more syntax.

Since the `.sir` file is the source of truth for imports, the visibility rule needs to be encoded there. The prototype-cloning walk above already skips `private` FuncOps, which gives you a reasonable default: everything non-private is importable unless you add an explicit export mechanism later.

### 3. Symbol Table Collision — Deduplicate at Insertion

If module A and module B both import module C, and a MAIN imports both A and B, the prototypes from C will be encountered twice. The `lookupSymbol` guard in `importPrototype` above handles this at the insertion point, but make sure deduplication is enforced at the final destination module, not per-import-invocation. A `llvm::StringSet<>` of already-imported symbol names threaded through the recursive import walk is a clean complement to the `lookupSymbol` check:

```cpp
llvm::StringSet<> importedSymbols;

// Before inserting, check both the set and the module's symbol table.
// The set catches in-progress duplicates before they hit the module.
```

### 4. Cycle Detection — Use a Two-State Visited Map

A simple `visited` boolean map will not catch all cycles. If `A → B → A`, and you only mark nodes as done *after* fully processing them, you'll start processing A a second time before detecting the cycle.

Use a two-state enum — **Visiting** (gray) and **Done** (black) — and mark a module as `Visiting` *before* recursing into its imports:

```cpp
enum class ImportState { Visiting, Done };
llvm::StringMap<ImportState> visited;

void processImports( llvm::StringRef moduleName, ... )
{
    auto [it, inserted] = visited.try_emplace( moduleName, ImportState::Visiting );
    if ( !inserted )
    {
        if ( it->second == ImportState::Visiting )
        {
            // Cycle detected — emit error with the module name
        }
        return; // Already done
    }

    // ... recurse into this module's own IMPORTs ...

    it->second = ImportState::Done;
}
```

This is the standard DFS gray/black coloring used in topological sort.

### 5. Timestamp Checking — Use `llvm::sys::fs::getLastModificationTime()`

The right LLVM API for mtime is:

```cpp
llvm::sys::TimePoint<> mtime;
llvm::sys::fs::getLastModificationTime( path, mtime );
```

Compare the `.sir` mtime against the `.silly` source mtime. If the source is newer, recompile.

One subtlety: if multiple source files are passed in a single compiler invocation, and `caller.silly` imports `callee.silly`, both appearing on the command line, you'd want to compile `callee` first without requiring a disk round-trip for the `.sir`. A topological sort of the input files based on their IMPORT graphs — done before any compilation begins — handles this cleanly and also catches cycles in the same-invocation case before any code generation starts.

### 6. Re-exported Declarations

The `importAllPrototypes` walk above skips `isExternal()` FuncOps (i.e., prototypes that were themselves imported into the `.sir` being read). This is intentional — you don't want to re-export another module's imports as if they were your own definitions. If you later want explicit re-export semantics (`EXPORT IMPORT foo` style), that's a separate feature.

---

## Suggested Implementation Order

1. **Phase 0**: Add `.mlirbc` write (`--emit-sir` flag) and read (`.sir`/`.sirbc` extension handling in `getInputType`) to the driver. Validate round-trip.
2. **Phase 1**: Implement the prototype-cloning walk and `importAllPrototypes`. Test with manually constructed `.sir` files (no IMPORT parsing yet).
3. **Phase 2**: Add `IMPORT` syntax to the parser/listener. Wire it to the source manager and prototype importer.
3. **Phase 2b**: Bridge: IMPORT sources have only prototypes to start with, different from implementation.  Require a different suffix (.sillymod), and then phase that out when ready to do the next steps (single sourcing implementation and interface at that point.)
4. **Phase 3**: Add timestamp checking and conditional recompilation.
5. **Phase 4**: Add cycle detection and multi-file topological sort.
6. **Phase 5**: Revisit visibility — decide if an explicit `EXPORT` keyword is warranted.
