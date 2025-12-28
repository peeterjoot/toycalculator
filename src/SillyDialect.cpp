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

// vim: et ts=4 sw=4
