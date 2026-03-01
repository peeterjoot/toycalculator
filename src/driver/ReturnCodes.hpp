/// @file    CompilationUnit.hpp
/// @author  Peeter Joot <peeterjoot@pm.me>
/// @brief   Silly compiler driver return codes.
///
#pragma once

namespace silly
{
    /// The numeric return codes for the silly driver
    enum class ReturnCodes : int
    {
        success,
        badExtensionError,
        badOption,
        directoryError,
        duplicateCUError,
        filenameParseError,
        ioError,
        linkError,
        loweringError,
        missingCUError,
        openError,
        parseError,
        tempCreationError,
        verifyError,
        LAST_ERROR_VALUE
    };
}    // namespace silly
