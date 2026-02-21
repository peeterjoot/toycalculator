/// @file    CompilationUnit.hpp
/// @author  Peeter Joot <peeterjoot@pm.me>
/// @brief   Optimization levels for the Silly compiler.
///
#pragma once

namespace silly
{
    /// Allowed optimization levels
    enum class OptLevel : int
    {
        O0,
        O1,
        O2,
        O3
    };
}

