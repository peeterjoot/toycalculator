/// @file LocationStack.hpp
/// @author Peeter Joot <peeterjoot@pm.me>
/// @brief MLIR location fusion helper code.
#pragma once
#include <mlir/IR/Builders.h>

namespace silly
{
    /// Location stack that should be fused for the current statement (declare, assignment, print, ...)
    ///
    /// Having the current-location state managed in an RAII fashion has the advantage of
    /// allowing automatic cleanup in error codepaths.
    ///
    /// There should only be one instance of LocationStack active at any point in time.
    class LocationStack
    {
       public:
        LocationStack( mlir::OpBuilder &b, mlir::Location loc );

        void push_back( mlir::Location loc );

        mlir::Location fuseLocations();

       private:
        mlir::OpBuilder &builder;

        /// locations that should be fused when the current statement processing is complete.
        llvm::SmallVector<mlir::Location, 4> locs{};
    };
}

// vim: et ts=4 sw=4
