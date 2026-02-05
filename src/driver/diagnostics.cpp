///
/// @file diagnostics.cpp
/// @author Peeter Joot <peeterjoot@pm.me>
/// @brief Helper functions for diagnostic output
///
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Location.h>

#include <format>
#include <fstream>
#include <string>

#include "diagnostics.hpp"

namespace silly
{
    void emitUserError( mlir::Location loc, const std::string& message, const std::string& funcName,
                        const std::string& sourceFile )
    {
        auto fileLoc = mlir::dyn_cast<mlir::FileLineColLoc>( loc );
        if ( !fileLoc )
        {
            llvm::errs() << std::format( "error: {}\n", message );
        }

        std::string filename = fileLoc.getFilename().str();
        unsigned line = fileLoc.getLine();
        unsigned col = fileLoc.getColumn();

        if ( funcName != ENTRY_SYMBOL_NAME )
        {
            llvm::errs() << std::format( "{}: In function ‘{}’:\n", filename, funcName );
        }

        // Print: filename:line:col: error: message
        llvm::errs() << std::format( "{}:{}:{}: error: {}\n", filename, line, col, message );

        // Try to read and display the source line
        if ( !sourceFile.empty() || !filename.empty() )
        {
            std::string path = sourceFile.empty() ? filename : sourceFile;

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
    }

}    // namespace silly
