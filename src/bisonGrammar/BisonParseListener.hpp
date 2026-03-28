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

        void enterDeclareStatement( const silly::TypeAndLoc& type, const silly::StringAndLoc& var,
                                    const silly::StringAndLoc& arraySize, const std::vector<silly::Expr>& initializers );

        void enterDeclareStatement( const silly::TypeAndLoc& type, const silly::StringAndLoc& var,
                                    const silly::StringAndLoc& arraySize );

        void enterDeclareStatementEmptyInit( const silly::TypeAndLoc& type, const silly::StringAndLoc& var,
                                             const silly::StringAndLoc& arraySize );

        void enterStringDeclareStatement( const silly::TypeAndLoc& type, const silly::StringAndLoc& var, const silly::StringAndLoc& arraySize,
                                          const std::string& init );

        void enterAssignmentStatement( const silly::Expr& var, const silly::Expr& rhs );

        void enterExitStatement( const silly::BisonParser::location_type& loc, const silly::Expr& var );

        void enterAbortStatement( const silly::BisonParser::location_type& loc );

        void enterGetStatement( const silly::BisonParser::location_type& bLoc, const silly::StringAndLoc& varName,
                                const silly::Expr& indexExpr );

        void enterImportStatement( const silly::BisonParser::location_type& bLoc, const silly::StringAndLoc& modName );

        void enterFunctionPrototype( const silly::BisonParser::location_type& funcLoc, const silly::StringAndLoc& id,
                                     const std::vector<silly::TypeAndName>& params, const silly::TypeAndLoc& returnType );

        void enterFunctionDefinition( const silly::BisonParser::location_type& funcLoc, const silly::StringAndLoc& name,
                                      const std::vector<silly::TypeAndName>& params, const silly::TypeAndLoc& returnType );

        void exitFunctionDefinition( const silly::BisonParser::location_type& funcLoc );

        void enterReturnStatement( const silly::BisonParser::location_type& bLoc, const silly::Expr& expr );

        void enterCallStatement( const silly::BisonParser::location_type& bLoc, const silly::StringAndLoc& id,
                                 const std::vector<silly::Expr>& args );

        void enterForStatement( const silly::BisonParser::location_type& bForLoc, const silly::TypeAndLoc& intType,
                                const silly::StringAndLoc& varId, const silly::Expr& start, const silly::Expr& stop,
                                const silly::Expr& step );

        void exitForStatement( const silly::BisonParser::location_type& bForLoc );

        void enterIfStatement( const silly::BisonParser::location_type& bLoc, const silly::Expr& predicate );

        void enterElifStatement( const silly::BisonParser::location_type& bLoc, const silly::Expr& predicate );

        void enterElseStatement( const silly::BisonParser::location_type& bLoc );

        void exitIfElifElseStatement( const silly::BisonParser::location_type& bLoc );

        void enterScopedStatements( const silly::BisonParser::location_type& bLoc );

        void exitScopedStatements();

        /// Called from parser on error
        void emitParseError( const silly::BisonParser::location_type& loc, const std::string& msg );

        yyscan_t getScanner();

        void setModule();

        void setPrintContinue();

        void setPrintError();

       private:
        mlir::Location getLocation( const silly::BisonParser::location_type& bloc );

        LocPairs getLocations( const silly::BisonParser::location_type& bloc, bool unique );

        mlir::Type declarationType( mlir::Location loc, const Types type );

        void declarationHelper( mlir::Location tLoc, const silly::StringAndLoc& var, const silly::StringAndLoc& arraySize,
                                mlir::Type ty, bool hasInit, const std::vector<silly::Expr>& initializerLiterals,
                                LocationStack& ls );

        void functionHelper( const silly::BisonParser::location_type& funcLoc, const silly::StringAndLoc& id,
                             const std::vector<silly::TypeAndName>& params, const silly::TypeAndLoc& returnType,
                             bool isDeclaration );

        mlir::Value parseIntermediate( mlir::Type ty, const silly::Expr& parg, LocationStack& ls );

        /// Calls parseIntermediate and does a final type conversion to the supplied type
        mlir::Value parseExpression( mlir::Type ty, const silly::Expr& parg, LocationStack& ls );

        template <class ExprVector>
        mlir::Value generateCall( const std::string& name, const ExprVector& args, mlir::Location loc,
                                  bool isCallStatement );

        mlir::Value parseReturnExpression( mlir::Location loc, const silly::Expr& expr, LocationStack& ls );

        yyscan_t scanner{};

        bool hasExplicitExit{};
        bool hasPrintContinue{};
        bool hasPrintError{};
    };
}    // namespace silly
