/**
 * @file    ToyDialect.cpp
 * @author  Peeter Joot <peeterjoot@pm.me>
 * @brief   Includes the source headers generated from ToyDialect.td
 */
#include "ToyDialect.hpp"

#define GET_OP_CLASSES
#include "ToyDialect.cpp.inc"

// Include generated dialect definitions
#include "ToyDialectDefs.cpp.inc"

namespace toy
{
    void ToyDialect::initialize()
    {
        addOperations<
#define GET_OP_LIST
#include "ToyDialect.cpp.inc"
            >();
    }
}    // namespace toy

// vim: et ts=4 sw=4
