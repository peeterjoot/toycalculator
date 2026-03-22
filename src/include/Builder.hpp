/// @file Builder.hpp
/// @author Peeter Joot <peeterjoot@pm.me>
/// @brief Grammar agnostic MLIR builder helper code for the silly compiler.

#pragma once

#include <llvm/ADT/SmallString.h>
#include <mlir/IR/Builders.h>

#include <string>
#include <vector>

#include "MlirTypeCache.hpp"
#include "ParserPerFunctionState.hpp"
#include "SillyDialect.hpp"

namespace silly
{
    class SourceManager;
    class DriverState;
    class LocationStack;

    /// Start and end locations associated with parser context.
    using LocPairs = std::pair<mlir::Location, mlir::Location>;

    class Builder
    {
       public:
        /// Initializes function state.
        ///
        /// - Records parameter Values, and creates each parameter DebugNameOp.
        /// - set the current function name, and squirrel away the funcOp for lookup.
        /// - Creates initial dummy DebugScopeOp placeholder.
        ///
        void createScope( mlir::Location startLoc, mlir::func::FuncOp funcOp, const std::string &funcName,
                          const std::vector<std::string> &paramNames );

        /// Returns a reference to the functionStateMap entry for funcName.
        ///
        /// Create that functionStateMap entry for funcName if it doesn't exist.
        ParserPerFunctionState &funcState( const std::string &funcName );

        /// @param floc FuncOp location.  This should be a fused location, or overwritten with a fused location later.
        /// @param sloc1 DebugNameOp location for parameters.
        void createMain( mlir::Location floc, mlir::Location sloc1 );

        void createMainExit( mlir::Location loc );

        /// Emit a user-friendly error message in GCC/Clang style
        ///
        /// (calls emitError)
        void emitUserError( mlir::Location loc, const std::string &message, const std::string &funcName )
        {
            emitError( loc, message, funcName, false );
        }

        /// Emit an internal error message, including the location in the compiler source where the error occured.
        ///
        /// (calls emitError)
        void emitInternalError( mlir::Location loc, const char *compilerfile, unsigned compilerline,
                                const char *compilerfunc, const std::string &message,
                                const std::string &programFuncName );

        /// Internal parse listener error message output.
        ///
        /// Show the file:line:col: error: message (colorized if desired.)
        ///
        /// errorCount is incremented as a side effect.
        void emitError( mlir::Location loc, const std::string &message, const std::string &funcName, bool internal );

        /// Construct a Value for a TRUE or FALSE boolean literal string
        mlir::Value parseBoolean( mlir::Location loc, const std::string &s, LocationStack &ls );

        /// Construct a Value for an integer literal string
        mlir::Value parseInteger( mlir::Location loc, int width, const std::string &s, LocationStack &ls );

        /// Construct a Value for a floating point literal string
        mlir::Value parseFloat( mlir::Location loc, mlir::FloatType ty, const std::string &s, LocationStack &ls );

        /// Strip double quotes off of a string, and build a string literal op for it
        silly::StringLiteralOp buildStringLiteral( mlir::Location loc, const std::string &input, LocationStack &ls );

        /// Registers a variable declaration in the current scope.
        void registerDeclaration( mlir::Location loc, const std::string &varName, mlir::Type ty, mlir::Location aLoc,
                                  const std::string &arrayBounds, bool haveInitializers,
                                  std::vector<mlir::Value> &initializers, LocationStack &ls );

        mlir::Value variableToValue( mlir::Location loc, const std::string &varName, mlir::Value iValue,
                                     mlir::Location iLoc, LocationStack &ls );

        /// Looks up DeclareOp for a variable.
        silly::DeclareOp lookupDeclareForVar( mlir::Location loc, const std::string &varName );

        /// Casts index value to index type.
        mlir::Value indexTypeCast( mlir::Location loc, mlir::Value val, LocationStack &ls );

        /// Casts value to desired type if needed.
        ///
        /// This is adapted from AssignOpLowering, but uses arith dialect operations instead of LLVM dialect.
        mlir::Value castOpIfRequired( mlir::Location loc, mlir::Value value, mlir::Type desiredType,
                                      LocationStack &ls );

        /// Handle assignment processing, given the current var-name and index (if appropriate.)
        void processAssignment( mlir::Location loc, mlir::Value resultValue, const std::string &currentVarName,
                                mlir::Value currentIndexExpr, LocationStack &ls );

        /// Lookup in per-function state, whether a variable has been declared
        bool isVariableDeclared( const std::string &varName );

        /// Emits silly::ReturnOp (or exit equivalent) with optional value.
        void processReturnLike( mlir::Location loc, mlir::Value returnValue,
                                LocationStack &ls );

        mlir::Type findReturnType( );

        /// Create a silly::ArithBinOp
        mlir::Value createBinaryArith( mlir::Location loc, silly::ArithBinOpKind what, mlir::Type ty,
                                       mlir::Value lhs, mlir::Value rhs, LocationStack &ls );

        /// Create a silly::CmpBinOp
        mlir::Value createBinaryCmp( mlir::Location loc, silly::CmpBinOpKind what, mlir::Value lhs,
                                     mlir::Value rhs, LocationStack &ls );

        enum class UnaryOp : uint32_t
        {
            Undefined,
            Negate,
            Plus,
            Not
        };

        mlir::Value makeUnaryExpression( mlir::Location loc, mlir::Value value, UnaryOp op, LocationStack & ls );

       protected:
        Builder( silly::SourceManager &s, const std::string &filename );

        /// back reference to the owning SourceManager (used for IMPORT module lookup)
        silly::SourceManager &sm;

        /// Compilation command line options and other stuff
        DriverState &driverState;

        /// The path to the source being processed.
        const std::string &sourceFile;

        /// Context for all the loaded dialects.
        mlir::MLIRContext *ctx;

        /// MLIR builder.
        mlir::OpBuilder builder;

        /// Top-level module.
        mlir::OwningOpRef<mlir::ModuleOp> rmod;

        /// mlir::Type values that will be used repeatedly
        MlirTypeCache typ;

        /// Saved insertion point for main.
        mlir::OpBuilder::InsertPoint mainIP{};

        /// Current function name.
        std::string currentFuncName;

        /// Per-function state map.
        std::unordered_map<std::string, std::unique_ptr<ParserPerFunctionState>> functionStateMap;

        /// Syntax errors detected.  Return a nullptr Module if this is non-zero.
        int errorCount{};

        /// By default silly programs have a main ('MAIN;' at start is implied.)  If, instead
        /// of MAIN (implicit or explicit), a 'MODULE;' is specified, that source may have
        /// only FUNCTIONs.
        bool isModule{};
    };
}    // namespace silly

// vim: et ts=4 sw=4
