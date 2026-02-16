///
/// @file DialectContext.cpp
/// @author Peeter Joot <peeterjoot@pm.me>
/// @brief Registration glue for the silly dialect dependencies
///
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>

#include "DialectContext.hpp"
#include "SillyDialect.hpp"

namespace silly
{
    DialectContext::DialectContext()
    {
        context.getOrLoadDialect<silly::SillyDialect>();
        context.getOrLoadDialect<mlir::func::FuncDialect>();
        context.getOrLoadDialect<mlir::arith::ArithDialect>();
        context.getOrLoadDialect<mlir::memref::MemRefDialect>();
        context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
        context.getOrLoadDialect<mlir::scf::SCFDialect>();
    }
}    // namespace silly
