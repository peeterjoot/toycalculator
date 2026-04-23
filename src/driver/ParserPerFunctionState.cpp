/// @file    ParserPerFunctionState.cpp
/// @author  Peeter Joot <peeterjoot@pm.me>
/// @brief   Grammar agnostic builder infrastructure
#include "ParserPerFunctionState.hpp"

namespace silly
{
    //--------------------------------------------------------------------------
    // ParserPerFunctionState members
    ParserPerFunctionState::ParserPerFunctionState()
        : lastDeclareOp{}, op{}, inductionVariables{}, parameters{}, variables{}, insertionPointStates{}
    {
    }

    mlir::Value ParserPerFunctionState::searchForInduction( const std::string& varName )
    {
        mlir::Value r{};

        for ( auto& p : inductionVariables )
        {
            if ( p.first == varName )
            {
                r = p.second;
                break;
            }
        }

        return r;
    }

    void ParserPerFunctionState::pushInductionVariable( const std::string& varName, mlir::Value i )
    {
        inductionVariables.emplace_back( varName, i );
    }

    bool ParserPerFunctionState::popInductionVariable()
    {
        if ( inductionVariables.size() )
        {
            inductionVariables.pop_back();
            return false;
        }
        else
        {
            return true; // error.
        }
    }

    mlir::Value ParserPerFunctionState::searchForParameter( const std::string& varName )
    {
        auto it = parameters.find( varName );
        return ( it != parameters.end() ) ? it->second : nullptr;
    }

    mlir::Value ParserPerFunctionState::searchForVariable( const std::string& varName )
    {
        for ( auto& vars : variables )
        {
            auto it = vars.find( varName );

            if ( it != vars.end() )
            {
                return it->second;
            }
        }

        return nullptr;
    }

    void ParserPerFunctionState::recordParameterValue( const std::string& varName, mlir::Value i )
    {
        parameters[varName] = i;
    }

    void ParserPerFunctionState::recordVariableValue( const std::string& varName, mlir::Value i )
    {
        if ( variables.size() == 0 )
        {
            variables.push_back( {} );
        }

        variables.back()[varName] = i;
    }

    void ParserPerFunctionState::createVariableLookupScope()
    {
        variables.push_back( {} );
    }

    void ParserPerFunctionState::destroyVariableLookupScope()
    {
        if ( variables.size() )
        {
            variables.pop_back();
        }
    }

    InsertionPointState& ParserPerFunctionState::createNewInsertionPointState()
    {
        insertionPointStates.push_back( {} );
        return insertionPointStates.back();
    }

    InsertionPointState& ParserPerFunctionState::currentInsertionPointState()
    {
        return insertionPointStates.back();
    }

    void ParserPerFunctionState::popInsertionPointState( mlir::OpBuilder& builder )
    {
        builder.setInsertionPointToStart( insertionPointStates.back().mergeBlock );
        insertionPointStates.pop_back();
    }

    bool ParserPerFunctionState::haveInsertionPointState()
    {
        return ( insertionPointStates.size() != 0 );
    }
}    // namespace silly

// vim: et ts=4 sw=4
