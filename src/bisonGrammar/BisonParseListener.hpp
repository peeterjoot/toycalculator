///
/// @file    BisonParseListener.hpp
/// @author  Peeter Joot <peeterjoot@pm.me>
/// @brief   Bison based experimental parse tree listener and MLIR builder.
///
#pragma once
#include <memory>
#include <string>

#include "silly_bison.tab.hh"

// Forward declare the flex scanner type
typedef void* yyscan_t;

namespace silly
{
    class BisonParseListener
    {
       public:
        BisonParseListener( const std::string& filename ) : filename{ filename }
        {
        }

        ~BisonParseListener() = default;

        /// Parse the given file, returns true on success
        bool parse();

        /// Called from parser action for PRINT statement
        void emitPrint( int value, const silly::BisonParser::location_type& printLoc,
                        const silly::BisonParser::location_type& valueLoc );

        /// Called from parser on error
        void emitError( const silly::BisonParser::location_type& loc, const std::string& msg );

        /// Current source filename (for location info)
        std::string filename;

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
