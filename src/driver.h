/**
 * @file    driver.h
 * @author  Peeter Joot <peeterjoot@pm.me>
 * @brief   State to pass between driver and lowering pass.
 */
#if !defined __Toy_driver_h_is_included
#define __Toy_driver_h_is_included

#include <string>

#pragma once

namespace toy
{
    struct driverState
    {
        bool isOptimized;
        bool wantDebug;
        std::string filename;
    };
}

#endif

// vim: et ts=4 sw=4
