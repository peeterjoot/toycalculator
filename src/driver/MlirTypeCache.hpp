///
/// @file    MlirTypeCache.hpp
/// @author  Peeter Joot <peeterjoot@pm.me>
/// @brief   Squirrel away some frequently used mlir::Type values
///
#pragma once

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>

namespace silly
{
    /// convenience types, so that get calls aren't needed all over the place
    struct MlirTypeCache
    {
        /// Initialize all the cached types.
        void initialize( mlir::OpBuilder &builder, mlir::MLIRContext *ctx );

        /// i1 type.
        mlir::IntegerType i1;

        /// (signed) i8 type.
        mlir::IntegerType i8;

        /// (signed) i16 type.
        mlir::IntegerType i16;

        /// (signed) i32 type.
        mlir::IntegerType i32;

        /// (signed) i64 type.
        mlir::IntegerType i64;

        /// f32 type.
        mlir::FloatType f32;

        /// f64 type.
        mlir::FloatType f64;

        /// LLVM pointer type.
        mlir::LLVM::LLVMPointerType ptr;

        /// LLVM void type.
        mlir::LLVM::LLVMVoidType voidT;
    };
}
