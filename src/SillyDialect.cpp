///
/// @file    SillyDialect.cpp
/// @author  Peeter Joot <peeterjoot@pm.me>
/// @brief   Includes the source headers generated from SillyDialect.td
///
#include "SillyDialect.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/DialectImplementation.h>

//#define GET_OP_CLASSES
//#include "SillyDialect.hpp.inc"

// Pull in generated op method bodies, adaptors, verify(), fold(), etc.
#define GET_OP_CLASSES
#include "SillyDialect.cpp.inc"

// Pull in generated type method bodies (parse, print, etc. if any)
#define GET_TYPEDEF_CLASSES
#include "SillyTypes.cpp.inc"

using namespace mlir;

namespace silly
{

    void SillyDialect::initialize()
    {
        // Register types
        addTypes<
#define GET_TYPEDEF_LIST
#include "SillyTypes.cpp.inc"
            >();

        // Register operations
        addOperations<
#define GET_OP_LIST
#include "SillyDialect.cpp.inc"
            >();
    }
}    // namespace silly

#include "SillyDialectDefs.cpp.inc"

// vim: et ts=4 sw=4
