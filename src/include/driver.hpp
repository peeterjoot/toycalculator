/// @file    driver.hpp
/// @author  Peeter Joot <peeterjoot@pm.me>
/// @brief   State to pass between driver and lowering pass.
#pragma once

#include <string>
#include <cstdint>

/// Name of the silly compiler driver, and used in llvm.ident and DICompileUnitAttr
#define COMPILER_NAME "silly"

namespace silly
{
    /// State to pass from the driver to parser/builder/lowering
    struct DriverState
    {
        /// True if not OptLevel::O0
        bool isOptimized{};

        /// True if -g is passed.
        bool wantDebug{};

        /// True for color error messages (when output is a terminal.)
        bool colorErrors{};

        /// Source file name passed to the driver.
        std::string filename{};

        /// Numeric --init-fill value if specified (zero otherwise.)
        uint8_t fillValue{};

        bool needsMathLib{};
    };

}    // namespace silly

// vim: et ts=4 sw=4
