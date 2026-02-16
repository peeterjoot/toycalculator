///
/// @file    MlirTypeCache.cpp
/// @author  Peeter Joot <peeterjoot@pm.me>
/// @brief   Squirrel away some frequently used mlir::Type values
///
#include "MlirTypeCache.hpp"

namespace silly
{
    void MlirTypeCache::initialize( mlir::OpBuilder &builder, mlir::MLIRContext *ctx )
    {
        i1 = builder.getI1Type();
        i8 = builder.getI8Type();
        i16 = builder.getI16Type();
        i32 = builder.getI32Type();
        i64 = builder.getI64Type();

        f32 = builder.getF32Type();
        f64 = builder.getF64Type();

        voidT = mlir::LLVM::LLVMVoidType::get( ctx );
        ptr = mlir::LLVM::LLVMPointerType::get( ctx );
    }
}
