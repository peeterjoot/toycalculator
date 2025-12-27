/// @file    ToyExceptions.hpp
/// @author  Peeter Joot <peeterjoot@pm.me>
/// @brief   Exception handling classes for the toy compiler.
#pragma once

#include <exception>
#include <format>

namespace toy
{
    /// An exception class for internal errors that builds a file:line:func triplet along with the reason.
    class exception_with_context : public std::exception
    {
       public:

        /// Example usage:
        ///
        /// ```
        /// throw exception_with_context(__FILE__, __LINE__, __func__, "Blah blah blah");
        /// ```
        exception_with_context( const char* file, int line, const char* func, const std::string& imessage )
        {
            message = std::format( "{}:{}:{}: {}", file, line, func, imessage );
        }

        /// Fetch the message text from the throw point.
        const char* what() const noexcept override
        {
            return message.c_str();
        }

       private:
        std::string message;
    };
}    // namespace toy

// vim: et ts=4 sw=4
