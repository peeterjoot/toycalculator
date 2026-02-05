///
/// @file diagnostics.hpp
/// @author Peeter Joot <peeterjoot@pm.me>
/// @brief Helper functions for diagnostic output
///
#pragma once

#include <string>

namespace mlir
{
    class Location;
}

#define ENTRY_SYMBOL_NAME "main"

namespace silly
{
    /// Emit a user-friendly error message in GCC/Clang style
    void emitUserError( mlir::Location loc, const std::string& message, const std::string& funcName,
                        const std::string& sourceFile );
}    // namespace silly
