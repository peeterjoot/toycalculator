///
/// @file    BisonParseListener.hpp
/// @author  Peeter Joot <peeterjoot@pm.me>
/// @brief   Bison based experimental parse tree listener and MLIR builder.
///
#pragma once

#include <memory>
#include <string>

#include "silly_bison.tab.hh"
#include "Builder.hpp"

// Forward declare the flex scanner type
typedef void* yyscan_t;

namespace silly
{
    class BisonParseListener : public Builder
    {
       public:
        BisonParseListener( silly::SourceManager &s, const std::string& filename ) : Builder{s, filename}
        {
        }

        ~BisonParseListener() = default;

        /// Parse the given file, and build the MLIR module for it
        mlir::OwningOpRef<mlir::ModuleOp> run();

        /// Called from parser action for PRINT statement
        void emitPrint( int value, const silly::BisonParser::location_type& printLoc,
                        const silly::BisonParser::location_type& valueLoc );

        /// Called from parser on error
        void emitError( const silly::BisonParser::location_type& loc, const std::string& msg );

        /// Error count
        int errorCount{};

        yyscan_t getScanner()
        {
            return scanner;
        }

       private:
        yyscan_t scanner{};
    };
}    // namespace silly
