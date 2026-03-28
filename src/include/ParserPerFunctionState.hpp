///
/// @file ParserPerFunctionState.hpp
/// @author Peeter Joot <peeterjoot@pm.me>
/// @brief Grammar agnostic MLIR builder helper stuff for the silly compiler.
///

#pragma once

#include <mlir/Dialect/Func/IR/FuncOps.h>

#include <cassert>

namespace silly
{
    /// Per-function state tracked during parsing.
    class ParserPerFunctionState
    {
       public:
        /// Default constructor
        ParserPerFunctionState();

        /// Getter for op, just to hide the casting
        mlir::func::FuncOp getFuncOp()
        {
            mlir::func::FuncOp funcOp{};
            if ( op )
            {
                funcOp = mlir::cast<mlir::func::FuncOp>( op );
            }

            return funcOp;
        }

        /// Setter for op, matching getFuncOp (which hides the casting)
        void setFuncOp( mlir::Operation *funcOp )
        {
            op = funcOp;
        }

        /// Search the inductionVariables stack for the named variable.
        ///
        /// This variable is pushed in enterForStatement, and popped in exitForStatement.
        mlir::Value searchForInduction( const std::string &varName );

        /// Add the mlir::Value for a named FOR loop variable to inductionVariables stack.
        void pushInductionVariable( const std::string &varName, mlir::Value i );

        /// Remove the top-most name/value pair from the inductionVariables stack.
        void popInductionVariable();

        /// Search parameters for the named variable.
        mlir::Value searchForParameter( const std::string &varName );

        /// Search variables for the named variable.
        mlir::Value searchForVariable( const std::string &varName );

        /// Add the mlir::Value for a parameter variable to the parameter list.
        void recordParameterValue( const std::string &varName, mlir::Value i );

        /// Add the mlir::Value for a variable variable to the variable list.
        void recordVariableValue( const std::string &varName, mlir::Value i );

        /// Is there an insertion point stack yet for this function?
        bool haveInsertionPointStack();

        /// Add to the insertion point stack for this function.
        void pushToInsertionPointStack( mlir::Operation *op );

        /// Remove last insertion point from the stack for this function.
        void popFromInsertionPointStack( mlir::OpBuilder &builder );

        /// Location of the last declaration for this function
        ///
        /// Declarations will all be inserted back to back before the function body statements.
        mlir::Operation *getLastDeclared()
        {
            return lastDeclareOp;
        }

        /// Set this operation (a declaration) as the last one for this function.
        void setLastDeclared( mlir::Operation *op )
        {
            lastDeclareOp = op;
        }

        /// New variables will be visible only for this scope and later
        void createVariableLookupScope();

        /// Any variables that had been declared in the current scope will no longer be visible.
        void destroyVariableLookupScope();

        /// Increase the level for scope_begin/scope_end
        /// @retval return the new scope-level
        int incrementScopeLevel()
        {
            scopeLevel++;

            return scopeLevel;
        }

        /// Decrease the level for scope_begin/scope_end
        void decrementScopeLevel()
        {
            assert( scopeLevel );

            scopeLevel--;
        }

        /// Obtain the level for the current or next scope_begin/scope_end pair.
        int getScopeLevel()
        {
            return scopeLevel;
        }

        /// For the bison front end.  Squirrel away the fact that there was a return statement.
        void setHaveReturn()
        {
            haveReturn = true;
        }

        /// For the bison front end.  Was there a return statement.
        bool getHaveReturn()
        {
            return haveReturn;
        }

       private:
        /// The last silly::DeclareOp created for the current function.
        ///
        /// The next declaration in the function will be placed after this, and
        /// this point updated accordingly.
        mlir::Operation *lastDeclareOp{};

        /// Associated func::FuncOp.
        mlir::Operation *op{};

        /// Induction variable name to Value mapping type
        using ValueList = std::vector<std::pair<std::string, mlir::Value>>;

        /// Variable and parameter name to Value mapping type
        using ValueMap = std::unordered_map<std::string, mlir::Value>;

        /// FOR loop variable stack containing all such variables that are in scope.
        ValueList inductionVariables;

        /// Parameter name/value pairs.
        ValueMap parameters;

        /// Variable name/value pairs.
        std::vector<ValueMap> variables;

        /// Stack for scf.if/scf.for blocks.
        std::vector<mlir::Operation *> insertionPointStack;

        /// For ScopeBeginOp/ScopeEndOp -- the scope level param.
        int scopeLevel{};

        /// Bison FE only.
        bool haveReturn{};
    };
}    // namespace silly

// vim: et ts=4 sw=4
