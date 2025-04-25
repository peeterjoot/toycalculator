/**
 * @file    ToyExceptions.h
 * @author  Peeter Joot <peeterjoot@pm.me>
 * @brief   Exception handling classes for the toy compiler.
 *
 */
#if !defined __ToyExceptions_h_is_included
#define __ToyExceptions_h_is_included

#pragma once

#include <exception>

namespace toy
{
    class semantic_exception : public std::exception
    {
       public:
        semantic_exception()
        {
        }

        const char *what()
        {
            return "semantic error";
        }
    };
}    // namespace toy

#endif

// vim: et ts=4 sw=4
