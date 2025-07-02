/**
 * @file    driver.hpp
 * @author  Peeter Joot <peeterjoot@pm.me>
 * @brief   State to pass between driver and lowering pass.
 */
#pragma once

#include <string>

namespace toy
{
    struct driverState
    {
        bool isOptimized;
        bool wantDebug;
        std::string filename;
    };
}

// vim: et ts=4 sw=4
