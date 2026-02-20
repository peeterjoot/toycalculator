///
/// @file    DriverState.cpp
/// @author  Peeter Joot <peeterjoot@pm.me>
/// @brief   Counting and logging of errors.
///
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinAttributes.h>

#include <format>
#include <fstream>

#include "DriverState.hpp"

namespace silly
{
    //--------------------------------------------------------------------------
    // DriverState members

    void DriverState::emitInternalError( mlir::Location loc, const char *compilerfile, unsigned compilerline,
                                         const char *compilerfunc, const std::string &message,
                                         const std::string &programFuncName )
    {
        emitError( loc, std::format( "{}:{}:{}: {}", compilerfile, compilerline, compilerfunc, message ),
                   programFuncName, true );
    }

    void DriverState::emitError( mlir::Location loc, const std::string &message, const std::string &funcName,
                                 bool internal )
    {
        bool inColor = isatty( fileno( stderr ) ) && colorErrors;
        const char *RED = inColor ? "\033[1;31m" : "";
        const char *CYAN = inColor ? "\033[0;36m" : "";
        const char *RESET = inColor ? "\033[0m" : "";

        if ( internal && errorCount )
        {
            errorCount++;
            return;
        }

        static std::string lastFunc{};
        auto fileLoc = mlir::dyn_cast<mlir::FileLineColLoc>( loc );
        if ( !fileLoc )
        {
            llvm::errs() << std::format( "{}{}error: {}{}\n", RED, internal ? "internal " : "", RESET, message );
        }

        std::string sourcename = fileLoc.getFilename().str();
        unsigned line = fileLoc.getLine();
        unsigned col = fileLoc.getColumn();

        if ( ( funcName != "" ) && ( funcName != ENTRY_SYMBOL_NAME ) && ( funcName != lastFunc ) )
        {
            llvm::errs() << std::format( "{}: In function ‘{}’:\n", sourcename, funcName );
        }
        lastFunc = funcName;

        // Print: sourcename:line:col: error: message
        llvm::errs() << std::format( "{}{}:{}:{}: {}{}error: {}{}\n", CYAN, sourcename, line, col, RED,
                                     internal ? "internal " : "", RESET, message );

        // Try to read and display the source line
        if ( !sourcename.empty() || !sourcename.empty() )
        {
            std::string path = sourcename.empty() ? sourcename : sourcename;

            if ( std::ifstream file{ path } )
            {
                std::string currentLine;
                unsigned currentLineNum = 0;

                while ( std::getline( file, currentLine ) )
                {
                    currentLineNum++;
                    if ( currentLineNum == line )
                    {
                        /* Example output:
                            5 | FOR(INT32 i:(0,2)){PRINT i;}
                              |     ^
                         */
                        // `{:>{}}` - the `^` character, right-aligned to `col` width
                        llvm::errs() << std::format(
                            "{0:5} | {1}\n"
                            "      | {2:>{3}}\n",
                            line, currentLine, "^", col );
                        break;
                    }
                }
            }
        }

        errorCount++;
    }
}    // namespace silly
