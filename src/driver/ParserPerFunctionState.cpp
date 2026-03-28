/// @file    ParserPerFunctionState.cpp
/// @author  Peeter Joot <peeterjoot@pm.me>
/// @brief   Grammar agnostic builder infrastructure
#include "ParserPerFunctionState.hpp"

namespace silly
{
    //--------------------------------------------------------------------------
    // ParserPerFunctionState members
    ParserPerFunctionState::ParserPerFunctionState()
        : lastDeclareOp{}, op{}, inductionVariables{}, parameters{}, variables{}, insertionPointStack{}
    {
    }

    mlir::Value ParserPerFunctionState::searchForInduction( const std::string &varName )
    {
        mlir::Value r{};

        for ( auto &p : inductionVariables )
        {
            if ( p.first == varName )
            {
                r = p.second;
                break;
            }
        }

        return r;
    }

    void ParserPerFunctionState::pushInductionVariable( const std::string &varName, mlir::Value i )
    {
        inductionVariables.emplace_back( varName, i );
    }

    void ParserPerFunctionState::popInductionVariable()
    {
        inductionVariables.pop_back();
    }

    mlir::Value ParserPerFunctionState::searchForParameter( const std::string &varName )
    {
        auto it = parameters.find( varName );
        return ( it != parameters.end() ) ? it->second : nullptr;
    }

    mlir::Value ParserPerFunctionState::searchForVariable( const std::string &varName )
    {
        for ( auto &vars : variables )
        {
            auto it = vars.find( varName );

            if ( it != vars.end() )
            {
                return it->second;
            }
        }

        return nullptr;
    }

    void ParserPerFunctionState::recordParameterValue( const std::string &varName, mlir::Value i )
    {
        parameters[varName] = i;
    }

    void ParserPerFunctionState::recordVariableValue( const std::string &varName, mlir::Value i )
    {
        if ( variables.size() == 0 )
        {
            variables.push_back( {} );
        }

        variables.back()[varName] = i;
    }

    void ParserPerFunctionState::startScope( mlir::Value value )
    {
        variables.push_back( {} );
    }

    void ParserPerFunctionState::endScope()
    {
        if ( variables.size() )
        {
            variables.pop_back();
        }
    }

    void ParserPerFunctionState::pushToInsertionPointStack( mlir::Operation *op )
    {
        insertionPointStack.push_back( op );
    }

    void ParserPerFunctionState::popFromInsertionPointStack( mlir::OpBuilder &builder )
    {
        builder.setInsertionPointAfter( insertionPointStack.back() );
        insertionPointStack.pop_back();
    }

    bool ParserPerFunctionState::haveInsertionPointStack()
    {
        return ( insertionPointStack.size() != 0 );
    }
}

// vim: et ts=4 sw=4
