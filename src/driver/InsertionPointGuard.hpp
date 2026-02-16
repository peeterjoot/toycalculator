///
/// @file    InsertionPointGuard.hpp
/// @author  Peeter Joot <peeterjoot@pm.me>
/// @brief   This file implements a couple RAII insertion point save/restore helpers.
///
#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Builders.h>

namespace silly
{
    /// An RAII insertion point save and restoration, setting new IP to module start
    class ModuleInsertionPointGuard
    {
       public:
        /// Constructor, that sets the insertion point to the beginning of the module.
        ///
        /// Also saves the current insertion point.
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

    /// An RAII insertion point save and restore.
    ///
    /// FIXME: does something like this already exist?
    class InsertionPointGuard
    {
       public:
        /// Saves the current insertion point.
        InsertionPointGuard( mlir::OpBuilder& opBuilder )
            : builder{ opBuilder }, oldIP{ builder.saveInsertionPoint() }
        {
        }

        /// Destructor, that restores the original insertion point.
        ~InsertionPointGuard()
        {
            builder.restoreInsertionPoint( oldIP );
        }

       private:
        mlir::OpBuilder& builder;    ///< cache the builder for IP restoration.

        mlir::OpBuilder::InsertPoint oldIP;    ///< the old IP
    };
}
