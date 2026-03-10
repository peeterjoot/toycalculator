/// @file Builder.hpp
/// @author Peeter Joot <peeterjoot@pm.me>
/// @brief Grammar agnostic MLIR builder helper code for the silly compiler.

#pragma once

#include <llvm/ADT/SmallString.h>
#include <mlir/IR/Builders.h>

#include <string>
#include <vector>

#include "MlirTypeCache.hpp"
#include "ParserPerFunctionState.hpp"

namespace silly
{
    class SourceManager;
    class DriverState;

    /// Start and end locations associated with parser context.
    using LocPairs = std::pair<mlir::Location, mlir::Location>;

    class Builder
    {
       public:
        /// Initializes function state.
        ///
        /// - Records parameter Values, and creates each parameter DebugNameOp.
        /// - set the current function name, and squirrel away the funcOp for lookup.
        /// - Creates initial dummy DebugScopeOp placeholder.
        ///
        void createScope( mlir::Location startLoc, mlir::func::FuncOp funcOp,
                          const std::string &funcName, const std::vector<std::string> &paramNames );

        /// Returns a reference to the functionStateMap entry for funcName.
        ///
        /// Create that functionStateMap entry for funcName if it doesn't exist.
        ParserPerFunctionState &funcState( const std::string &funcName );

        /// @param floc FuncOp location.  This should be a fused location, or overwritten with a fused location later.
        /// @param sloc1 DebugNameOp location for parameters.
        void createMain( mlir::Location floc, mlir::Location sloc1 );

        void createMainExit( mlir::Location loc );

       protected:
        Builder( silly::SourceManager &s, const std::string &filename );

        /// back reference to the owning SourceManager (used for IMPORT module lookup)
        silly::SourceManager &sm;

        /// Compilation command line options and other stuff
        DriverState &driverState;

        /// The path to the source being processed.
        const std::string &sourceFile;

        /// Context for all the loaded dialects.
        mlir::MLIRContext *ctx;

        /// MLIR builder.
        mlir::OpBuilder builder;

        /// Top-level module.
        mlir::OwningOpRef<mlir::ModuleOp> rmod;

        /// mlir::Type values that will be used repeatedly
        MlirTypeCache typ;

        /// Saved insertion point for main.
        mlir::OpBuilder::InsertPoint mainIP{};

        /// Current function name.
        std::string currentFuncName;

        /// Per-function state map.
        std::unordered_map<std::string, std::unique_ptr<ParserPerFunctionState>> functionStateMap;

        /// Syntax errors detected.  Return a nullptr Module if this is non-zero.
        int errorCount{};

        /// By default silly programs have a main ('MAIN;' at start is implied.)  If, instead
        /// of MAIN (implicit or explicit), a 'MODULE;' is specified, that source may have
        /// only FUNCTIONs.
        bool isModule{};
    };
}    // namespace silly

// vim: et ts=4 sw=4
