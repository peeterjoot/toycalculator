/// @file    CompilationUnit.hpp
/// @author  Peeter Joot <peeterjoot@pm.me>
/// @brief   Silly compiler driver return codes.
///
/// FIXME: revisit direct exit as a FATAL mechanism -- will orphan stuff.
///
#pragma once

/// The numeric return codes for the silly driver
enum class ReturnCodes : int
{
    success,
    badExtensionError,
    directoryError,
    filenameParseError,
    ioError,
    linkError,
    loweringError,
    openError,
    parseError,
    tempCreationError,
    verifyError,
};


