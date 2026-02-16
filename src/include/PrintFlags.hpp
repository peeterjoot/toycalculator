///
/// @file PrintFlags.hpp
/// @brief print related types and enums, shared between builder, lowering and runtime.
/// @author  Peeter Joot <peeterjoot@pm.me>
///
#pragma once

#include <stdint.h>

namespace silly
{
    /// Bitmask flags for PRINT builtin and runtime function
    enum PrintFlags : uint32_t
    {
        PRINT_FLAGS_NONE = 0,
        PRINT_FLAGS_CONTINUE = 1U << 0,    //< set if a PRINT should omit a newline
        PRINT_FLAGS_ERROR = 1U << 1        //< set for PRINT to stderr, instead of stdout.
    };

    /// Variable type to print
    enum class PrintKind : uint32_t
    {
        UNKNOWN = 0,
        I64,
        F64,
        STRING
    };

    /// One PrintArg for each PRINT argument
    struct PrintArg
    {
        PrintKind kind;      ///< I64, F64, STRING, ...
        PrintFlags flags;    ///< newline, stdout/err selection, and perhaps eventually format flags
        int64_t i;           ///< value for I64 case (or bitcast double for F64 case), or for STRING, the string length.
        const char* ptr;     ///< only used for STRING.
    };
}    // namespace silly
