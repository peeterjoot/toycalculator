///
/// @file    helper.hpp
/// @author  Peeter Joot <peeterjoot@pm.me>
/// @brief   Some generic MLIR helper functions
///
#pragma once

#include <mlir/IR/Location.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>

namespace silly
{
    /// Assuming that a Location is actually a FileLineColLoc, cast it and return it as so.
    ///
    /// Will assert if this is not the case.
    mlir::FileLineColLoc locationToFLCLoc( mlir::Location loc );

    /// Find the mlir::func::FuncOp that contains the provided op.
    mlir::func::FuncOp getEnclosingFuncOp( mlir::Operation* op );

    /// Looks up the enclosing function name for an operation.
    std::string lookupFuncNameForOp( mlir::Operation* op );

    /// Formats location for error messages.
    std::string formatLocation( mlir::Location loc );

    /// Figure out the bigger of two types for implicit cast-like purposes
    mlir::Type biggestTypeOf( mlir::Type ty1, mlir::Type ty2 );

    /// While mlir::Type is streamable in << output, here's a little shim for use in string based dumps.
    std::string mlirTypeToString( mlir::Type t );
}
