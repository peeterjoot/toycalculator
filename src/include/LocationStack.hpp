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
        /// Construct a location stack
        LocationStack( mlir::OpBuilder& b, mlir::Location loc );

        /// Push another location to the back of the location stack
        void push_back( mlir::Location loc );

        /// Fuse multiple locations or return a single location unchanged.
        mlir::Location fuseLocations();

       private:
        /// Cached reference to the builder that we need to fuse locations.
        mlir::OpBuilder& builder;

        /// locations that should be fused when the current statement processing is complete.
        llvm::SmallVector<mlir::Location, 4> locs{};
    };
}    // namespace silly

// vim: et ts=4 sw=4
