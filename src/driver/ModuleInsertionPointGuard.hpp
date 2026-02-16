///
/// @file    ModuleInsertionPointGuard.hpp
/// @author  Peeter Joot <peeterjoot@pm.me>
/// @brief   This file implements a RAII insertion point save/restore helper to switch to start of module scope.
///
#pragma once

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>

namespace silly
{
    /// RAII helper to temporarily set insertion point to the start of a module body.
    /// Uses upstream mlir::OpBuilder::InsertionGuard under the hood for restore.
    class ModuleInsertionPointGuard
    {
       public:
        /// @brief Constructor, that sets the insertion point to the beginning of the module, saving the current IP.
        ///
        /// @param mod Module whose body to insert into
        /// @param builder Builder to guard
        ModuleInsertionPointGuard( mlir::ModuleOp& mod, mlir::OpBuilder& builder ) : guard( builder )
        {    // saves old IP
            builder.setInsertionPointToStart( mod.getBody() );
        }

       private:
        mlir::OpBuilder::InsertionGuard guard;    ///< restores IP on destruction
    };
}    // namespace silly
