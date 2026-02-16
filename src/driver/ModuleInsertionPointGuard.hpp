///
/// @file    ModuleInsertionPointGuard.hpp
/// @author  Peeter Joot <peeterjoot@pm.me>
/// @brief   This file implements a RAII insertion point save/restore helpers to switch to start of module scope.
///
#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Builders.h>

namespace silly
{
    /// @brief An RAII insertion point save and restoration, setting new IP to module start
    ///
    /// This is a RAII helper that temporarily sets the insertion point to the start of a module body
    /// and restores the original insertion point on scope exit.
    class ModuleInsertionPointGuard
    {
       public:
        /// @brief Constructor, that sets the insertion point to the beginning of the module, saving the current IP.
        ///
        /// @param mod Module whose body to insert into
        /// @param opBuilder Builder to guard
        ModuleInsertionPointGuard( mlir::ModuleOp& mod, mlir::OpBuilder& opBuilder )
            : builder{ opBuilder }, oldIP{ builder.saveInsertionPoint() }
        {
            builder.setInsertionPointToStart( mod.getBody() );
        }

        /// Destructor, that restores the original insertion point.
        ~ModuleInsertionPointGuard()
        {
            builder.restoreInsertionPoint( oldIP );
        }

       private:
        mlir::OpBuilder& builder;    ///< cache the builder for IP restoration.

        mlir::OpBuilder::InsertPoint oldIP;    ///< the old IP
    };
}
