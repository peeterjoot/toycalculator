/// @file Builder.hpp
/// @author Peeter Joot <peeterjoot@pm.me>
/// @brief Grammar agnostic MLIR builder helper code for the silly compiler.

#pragma once

#include <mlir/IR/Builders.h>
#include <llvm/ADT/SmallString.h>

#include <string>

#include "MlirTypeCache.hpp"
#include "ParserPerFunctionState.hpp"

namespace silly
{
    class SourceManager;
    class DriverState;

    class Builder
    {
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
