///
/// @file    SillyDialect.cpp
/// @author  Peeter Joot <peeterjoot@pm.me>
/// @brief   Includes the source headers generated from SillyDialect.td
///
#include "SillyDialect.hpp"

#define GET_OP_CLASSES
#include "SillyDialect.cpp.inc"

// Include generated dialect definitions
#include "SillyDialectDefs.cpp.inc"

namespace silly
{
    void SillyDialect::initialize()
    {
        addOperations<
#define GET_OP_LIST
#include "SillyDialect.cpp.inc"
            >();
    }
}    // namespace silly

void SillyDialect::initialize() {
    addTypes<
#define GET_TYPEDEF_LIST
#include "SillyTypes.cpp.inc"
    >();

    addOperations<
#define GET_OP_LIST
#include "SillyDialect.cpp.inc"
    >();
}

// vim: et ts=4 sw=4
