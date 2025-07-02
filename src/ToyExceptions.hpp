/**
 * @file    ToyExceptions.hpp
 * @author  Peeter Joot <peeterjoot@pm.me>
 * @brief   Exception handling classes for the toy compiler.
 */
#pragma once

#include <exception>
#include <format>

namespace toy
{
    class exception_with_context : public std::exception
    {
       public:
        exception_with_context( const char* file, int line, const char* func, const std::string& imessage )
        {
            message = std::format( "{}:{}:{}: {}", file, line, func, imessage );
        }

        const char* what() const noexcept override
        {
            return message.c_str();
        }

       private:
        std::string message;
    };
}    // namespace toy

// vim: et ts=4 sw=4
