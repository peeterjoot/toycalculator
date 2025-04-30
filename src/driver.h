/**
 * @file    driver.h
 * @author  Peeter Joot <peeterjoot@pm.me>
 * @brief   State to pass between driver and lowering pass.
 */
#if !defined __PJ_driver_h_is_included
#define __PJ_driver_h_is_included

#pragma once

namespace toy
{
    struct driverState
    {
        bool isOptimized;
    };
}

#endif
