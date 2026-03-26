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

        void enterDeclareStatement( const silly::Types& type, const std::string& varName,
                                    const std::string& arraySizeString, const std::vector<silly::Expr>& initializers,
                                    const silly::BisonParser::location_type& typeLoc,
                                    const silly::BisonParser::location_type& nameLoc,
                                    const silly::BisonParser::location_type& arrayLoc );

        void enterStringDeclareStatement( const std::string& varName,
                                          const std::string& arraySizeString, const std::string & init,
                                          const silly::BisonParser::location_type& typeLoc,
                                          const silly::BisonParser::location_type& nameLoc,
                                          const silly::BisonParser::location_type& arrayLoc );

        void enterDeclareStatementWithEmptyInit( const silly::Types& type, const std::string& varName,
                                                 const std::string& arraySizeString,
                                                 const silly::BisonParser::location_type& typeLoc,
                                                 const silly::BisonParser::location_type& nameLoc,
                                                 const silly::BisonParser::location_type& arrayLoc );

        void enterDeclareStatement( const silly::Types& type, const std::string& varName,
                                    const std::string& arraySizeString,
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

        void enterCallStatement( const std::string& name, const std::vector<silly::Expr>& args,
                                 const silly::BisonParser::location_type& bLoc );

        void enterForStatement( const silly::BisonParser::location_type& bForLoc, const silly::Types& intType,
                                const silly::BisonParser::location_type& bVarLoc, const std::string& varName,
                                const silly::Expr& start, const silly::Expr& stop, const silly::Expr& step );

        void exitForStatement( const silly::BisonParser::location_type& bForLoc );

        void enterIfStatement( const silly::BisonParser::location_type& bLoc, const silly::Expr& predicate );

        void enterElifStatement( const silly::BisonParser::location_type& bLoc, const silly::Expr& predicate );

        void enterElseStatement( const silly::BisonParser::location_type& bLoc );

        void exitIfElifElseStatement( const silly::BisonParser::location_type& bLoc );

        void enterScopedStatements( const silly::BisonParser::location_type& bLoc );

        void exitScopedStatements( );

        /// Called from parser on error
        void emitParseError( const silly::BisonParser::location_type& loc, const std::string& msg );

        yyscan_t getScanner();

        void setModule();

        void setPrintContinue();

        void setPrintError();

       private:
        mlir::Location getLocation( const silly::BisonParser::location_type& bloc );

        LocPairs getLocations( const silly::BisonParser::location_type& bloc );

        mlir::Type declarationType( mlir::Location loc, const Types type );

        void declarationHelper( mlir::Location tLoc, mlir::Location aLoc, const std::string& varName,
                                const std::string& arraySizeString, mlir::Type ty, bool hasInit,
                                const std::vector<silly::Expr>& initializerLiterals, LocationStack& ls );

        void functionHelper( const std::string& name, const std::vector<silly::TypeAndName>& params,
                             const silly::Types& returnType, const silly::BisonParser::location_type& funcLoc,
                             bool isDeclaration );

        mlir::Value parseIntermediate( mlir::Type ty, const silly::Expr& parg, LocationStack& ls );

        /// Calls parseIntermediate and does a final type conversion to the supplied type
        mlir::Value parseExpression( mlir::Type ty, const silly::Expr& parg, LocationStack& ls );

        template <class ExprVector>
        mlir::Value generateCall( const std::string& name, const ExprVector& args, mlir::Location loc,
                                  bool isCallStatement );

        mlir::Value parseReturnExpression( mlir::Location loc, const silly::Expr& expr, LocationStack &ls );

        yyscan_t scanner{};

        bool hasExplicitExit{};
        bool hasPrintContinue{};
        bool hasPrintError{};
    };
}    // namespace silly
