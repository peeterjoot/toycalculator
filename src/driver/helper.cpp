///
/// @file    helper.hpp
/// @author  Peeter Joot <peeterjoot@pm.me>
/// @brief   Some generic MLIR helper functions
///
#include "helper.hpp"

namespace silly
{
    /// Assuming that a Location is actually a FileLineColLoc, cast it and return it as so.
    ///
    /// Will assert if this is not the case.
    mlir::FileLineColLoc locationToFLCLoc( mlir::Location loc )
    {
        // Cast Location to FileLineColLoc
        mlir::FileLineColLoc fileLineLoc = mlir::dyn_cast<mlir::FileLineColLoc>( loc );
        assert( fileLineLoc );

        return fileLineLoc;
    }

    /// Find the mlir::func::FuncOp that contains the provided op.
    mlir::func::FuncOp getEnclosingFuncOp( mlir::Operation* op )
    {
        while ( op )
        {
            if ( mlir::func::FuncOp funcOp = dyn_cast<mlir::func::FuncOp>( op ) )
            {
                return funcOp;
            }
            op = op->getParentOp();
        }
        return nullptr;
    }

    std::string lookupFuncNameForOp( mlir::Operation* op )
    {
        mlir::func::FuncOp funcOp = getEnclosingFuncOp( op );

        return funcOp.getSymName().str();
    }
}
