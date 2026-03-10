///
/// @file    BisonParseListener.hpp
/// @author  Peeter Joot <peeterjoot@pm.me>
/// @brief   Bison based experimental parse tree listener and MLIR builder.
///
#pragma once

#include <memory>
#include <string>

#include "Builder.hpp"
#include "location.hh"
#include "silly_bison.tab.hh"

// Forward declare the flex scanner type
typedef void* yyscan_t;

namespace silly
{
    class BisonParseListener : public Builder
    {
       public:
        BisonParseListener( silly::SourceManager& s, const std::string& filename );

        ~BisonParseListener() = default;

        /// Parse the given file, and build the MLIR module for it
        mlir::OwningOpRef<mlir::ModuleOp> run();

        /// Called from parser action for PRINT statement
        void emitPrint( int value, const silly::BisonParser::location_type& printLoc,
                        const silly::BisonParser::location_type& valueLoc );

        /// Called from parser on error
        void emitError( const silly::BisonParser::location_type& loc, const std::string& msg );

        void enter( const silly::BisonParser::location_type& loc );

        void exit( const silly::BisonParser::location_type& loc );

        /// Error count
        int errorCount{};

        yyscan_t getScanner();

       private:
        mlir::Location getLocation( const silly::BisonParser::location_type& bloc );

        LocPairs getLocations( const silly::BisonParser::location_type& bloc );

        yyscan_t scanner{};
    };
}    // namespace silly
