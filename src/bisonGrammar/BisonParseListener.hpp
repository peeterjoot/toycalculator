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
    /// Bison grammar parse tree walker for the silly language (incomplete.)
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
        void enterPrintStatement( const std::vector<silly::Expr>& args,
                                  const silly::BisonParser::location_type& printLoc );

        void enterIntDeclareStatement( const silly::Types& type, const std::string& varName,
                                       const std::string& arraySizeString,
                                       const std::vector<silly::Literal>& initializers,
                                       const silly::BisonParser::location_type& typeLoc,
                                       const silly::BisonParser::location_type& nameLoc,
                                       const silly::BisonParser::location_type& arrayLoc );

        void enterFloatDeclareStatement( const silly::Types& type, const std::string& varName,
                                         const std::string& arraySizeString,
                                         const std::vector<silly::Literal>& initializers,
                                         const silly::BisonParser::location_type& typeLoc,
                                         const silly::BisonParser::location_type& nameLoc,
                                         const silly::BisonParser::location_type& arrayLoc );

        void enterBoolDeclareStatement( const std::string& varName, const std::string& arraySizeString,
                                        const std::vector<silly::Literal>& initializers,
                                        const silly::BisonParser::location_type& typeLoc,
                                        const silly::BisonParser::location_type& nameLoc,
                                        const silly::BisonParser::location_type& arrayLoc );

        void enterAssignmentStatement( const silly::Expr& var, const silly::Expr& rhs,
                                       const silly::BisonParser::location_type& lhsLoc,
                                       const silly::BisonParser::location_type& rhsLoc );

        void enterExitStatement( const silly::BisonParser::location_type& loc, const silly::Expr& var );

        void enterExitStatement( const silly::BisonParser::location_type& loc );

        void enterAbortStatement( const silly::BisonParser::location_type& loc );

        void enterGetStatement( const silly::BisonParser::location_type& bLoc, const std::string& varName );

        void enterGetStatement( const silly::BisonParser::location_type& bLoc, const std::string& varName,
                                const silly::Expr& indexExpr );

        void enterImportStatement( const silly::BisonParser::location_type& bLoc, const std::string& modName );

        void enterFunctionPrototype( const std::string& name, const std::vector<silly::TypeAndName>& params,
                                     const silly::Types& returnType, const silly::BisonParser::location_type& funcLoc );

        void enterFunctionDefinition( const std::string& name, const std::vector<silly::TypeAndName>& params,
                                      const silly::Types& returnType,
                                      const silly::BisonParser::location_type& funcLoc );

        void exitFunctionDefinition();

        void enterReturnStatement( const silly::BisonParser::location_type& bLoc, const silly::Expr& expr );

        void enterReturnStatement( const silly::BisonParser::location_type& bLoc );

        void enterCallStatement( const std::string& name, const std::vector<silly::Expr>& args,
                                 const silly::BisonParser::location_type& bLoc );

        /// Called from parser on error
        void emitParseError( const silly::BisonParser::location_type& loc, const std::string& msg );

        yyscan_t getScanner();

        void setModule();

        void setPrintContinue();
        void setPrintError();
        void setDeclarationAssignment();
        void hasDeclarationHasInitializer();

       private:
        mlir::Location getLocation( const silly::BisonParser::location_type& bloc );

        LocPairs getLocations( const silly::BisonParser::location_type& bloc );

        mlir::Type declarationType( mlir::Location loc, const Types type );

        void declarationHelper( mlir::Location tLoc, mlir::Location aLoc, const std::string& varName,
                                const std::string& arraySizeString, mlir::Type ty, bool initIsDeclare, bool hasInit,
                                const std::vector<silly::Literal>& initializerLiterals, LocationStack& ls );

        void functionHelper( const std::string& name, const std::vector<silly::TypeAndName>& params,
                             const silly::Types& returnType, const silly::BisonParser::location_type& funcLoc,
                             bool isDeclaration );

        mlir::Value parseExpression( mlir::Location vLoc, mlir::Type ty, const silly::Expr& parg, LocationStack& ls );

        void generateCall( const std::string& name, const std::vector<silly::Expr>& args,
                           const silly::BisonParser::location_type& bLoc, bool isCallStatement );

        yyscan_t scanner{};

        bool hasExplicitExit{};
        bool hasPrintContinue{};
        bool hasPrintError{};
        bool declarationAssignmentInitialization{};
        bool declarationHasInitializer{};
    };
}    // namespace silly
