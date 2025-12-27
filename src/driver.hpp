/// @file    driver.hpp
/// @author  Peeter Joot <peeterjoot@pm.me>
/// @brief   State to pass between driver and lowering pass.
#pragma once

#include <string>

namespace toy
{
    /// State to pass from the driver to lowering
    struct driverState
    {
        /// True if not OptLevel::O0
        bool isOptimized;

        /// True if -g is passed.
        bool wantDebug;

        /// Source file name passed to the driver.
        std::string filename;
    };
}

// vim: et ts=4 sw=4
