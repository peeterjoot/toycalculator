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
#include "silly.tab.hh"

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

        void enterStartRule( const silly::BisonParser::location_type& loc );

        void exitStartRule( const silly::BisonParser::location_type& loc );

        /// Called from parser action for PRINT statement
        void enterPrintStatement( const std::vector<silly::PrintContextArgument>& args,
                                  const silly::BisonParser::location_type& printLoc );

        void enterIntDeclare( const std::string& typeName, const std::string& varName,
                              const std::string& arraySizeString, Literal initializer,
                              const silly::BisonParser::location_type& typeLoc,
                              const silly::BisonParser::location_type& nameLoc,
                              const silly::BisonParser::location_type& arrayLoc );

        void enterFloatDeclare( const std::string& typeName, const std::string& varName,
                                const std::string& arraySizeString, Literal initializer,
                                const silly::BisonParser::location_type& typeLoc,
                                const silly::BisonParser::location_type& nameLoc,
                                const silly::BisonParser::location_type& arrayLoc );

        void enterBoolDeclare( const std::string& varName, const std::string& arraySizeString, Literal initializer,
                               const silly::BisonParser::location_type& typeLoc,
                               const silly::BisonParser::location_type& nameLoc,
                               const silly::BisonParser::location_type& arrayLoc );

        /// Called from parser on error
        void emitParseError( const silly::BisonParser::location_type& loc, const std::string& msg );

        yyscan_t getScanner();

        void setModule();

        void setPrintContinue();
        void setPrintError();
        void setDeclarationAssignment();

       private:
        mlir::Location getLocation( const silly::BisonParser::location_type& bloc );

        LocPairs getLocations( const silly::BisonParser::location_type& bloc );

        mlir::Type integerDeclarationType( mlir::Location loc, const std::string& typeName );

        void declarationHelper( mlir::Location tLoc, mlir::Location aLoc, const std::string& varName,
                                const std::string& arraySizeString, mlir::Type ty, Literal initializer,
                                LocationStack& ls );
        yyscan_t scanner{};

        bool hasPrintContinue{};
        bool hasPrintError{};
        bool declarationAssignmentInitialization{};
    };
}    // namespace silly
