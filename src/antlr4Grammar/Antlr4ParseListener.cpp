///
/// @file    Antlr4ParseListener.cpp
/// @author  Peeter Joot <peeterjoot@pm.me>
/// @brief   altlr4 parse tree listener and MLIR builder.
///
#include <llvm/Support/Debug.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Types.h>

#include <format>
#include <fstream>
#include <string>

#include "Antlr4ParseListener.hpp"
#include "DriverState.hpp"
#include "ModuleInsertionPointGuard.hpp"
#include "ParserPerFunctionState.hpp"
#include "SillyDialect.hpp"
#include "SillyLexer.h"
#include "SourceManager.hpp"
#include "helper.hpp"

/// --debug- class for the parser
#define DEBUG_TYPE "silly-parser"

namespace silly
{
    //--------------------------------------------------------------------------
    // Antlr4ParseListener members
    //
    Antlr4ParseListener::Antlr4ParseListener( silly::SourceManager &s, const std::string &filename )
        : Builder( s, filename )
    {
    }

    mlir::OwningOpRef<mlir::ModuleOp> Antlr4ParseListener::run()
    {
        driverState.openFailed = false;

        std::ifstream inputStream;
        inputStream.open( sourceFile );
        if ( !inputStream.is_open() )
        {
            driverState.openFailed = true;
            return nullptr;
        }

        antlr4::ANTLRInputStream antlrInput( inputStream );
        SillyLexer lexer( &antlrInput );
        antlr4::CommonTokenStream tokens( &lexer );
        SillyParser parser( &tokens );

        // Remove default error listener and add Antlr4ParseListener for errors
        parser.removeErrorListeners();
        parser.addErrorListener( this );

        antlr4::tree::ParseTree *tree = parser.startRule();
        antlr4::tree::ParseTreeWalker::DEFAULT.walk( this, tree );

        if ( errorCount )
        {
            return nullptr;
        }

        return std::move( rmod );
    }


    inline LocPairs Antlr4ParseListener::getLocations( antlr4::ParserRuleContext *ctx, bool unique )
    {
        size_t startLine = 1;
        size_t startCol = 0;
        size_t endLine = 1;
        size_t endCol = 0;

        if ( ctx )
        {
            antlr4::Token *startToken = ctx->getStart();
            startLine = startToken->getLine();
            startCol = startToken->getCharPositionInLine();

            antlr4::Token *endToken = ctx->getStop();
            endLine = endToken->getLine();
            endCol = endToken->getCharPositionInLine();
        }

        if ( ( startLine == endLine ) and ( startCol == endCol ) and unique )
        {
            endCol++;
        }

        mlir::FileLineColLoc startLoc =
            mlir::FileLineColLoc::get( builder.getStringAttr( sourceFile ), startLine, startCol + 1 );
        mlir::FileLineColLoc endLoc =
            mlir::FileLineColLoc::get( builder.getStringAttr( sourceFile ), endLine, endCol + 1 );

        return { startLoc, endLoc };
    }

    inline mlir::Location Antlr4ParseListener::getStartLocation( antlr4::ParserRuleContext *ctx )
    {
        LocPairs locs = getLocations( ctx );

        return locs.first;
    }

    inline mlir::Location Antlr4ParseListener::getStopLocation( antlr4::ParserRuleContext *ctx )
    {
        LocPairs locs = getLocations( ctx );

        return locs.second;
    }

    inline mlir::Location Antlr4ParseListener::getTokenLocation( antlr4::Token *token )
    {
        assert( token );
        size_t line = token->getLine();
        size_t column = token->getCharPositionInLine() + 1;

        return mlir::FileLineColLoc::get( builder.getStringAttr( sourceFile ), line, column );
    }

    inline mlir::Location Antlr4ParseListener::getTerminalLocation( antlr4::tree::TerminalNode *node )
    {
        assert( node );
        antlr4::Token *token = node->getSymbol();

        return getTokenLocation( token );
    }

    inline mlir::Value Antlr4ParseListener::parseExpression( SillyParser::ExpressionContext *ctx, mlir::Type ty,
                                                             LocationStack &ls )
    {
        mlir::Location loc = getStartLocation( ctx );
        mlir::Value value{};

        if ( !ctx )
        {
            emitInternalError( loc, __FILE__, __LINE__, __func__, "no ExpressionContext", currentFuncName );
            return value;
        }

        value = parseLowest( ctx, ty, ls );
        if ( !value )
        {
            emitInternalError( loc, __FILE__, __LINE__, __func__, "parseLowest failed", currentFuncName );
            return value;
        }

        if ( ty )
        {
            value = createCastOpIfRequired( loc, value, ty, ls );
        }

        return value;
    }

    inline mlir::Value Antlr4ParseListener::parseLowest( antlr4::ParserRuleContext *ctx, mlir::Type ty,
                                                         LocationStack &ls )
    {
        SillyParser::ExprLowestContext *expr = dynamic_cast<SillyParser::ExprLowestContext *>( ctx );
        if ( !expr )
        {
            mlir::Location loc = getStartLocation( ctx );

            emitInternalError( loc, __FILE__, __LINE__, __func__,
                               std::format( "unexpected ExpressionContext alternative: {}", ctx->getText() ),
                               currentFuncName );
            return mlir::Value{};
        }

        SillyParser::BinaryExpressionLowestContext *lowest = expr->binaryExpressionLowest();

        return parseOr( lowest, ty, ls );
    }

    void Antlr4ParseListener::syntaxError( antlr4::Recognizer *recognizer, antlr4::Token *offendingSymbol, size_t line,
                                           size_t charPositionInLine, const std::string &msg, std::exception_ptr e )
    {
        if ( offendingSymbol )
        {
            mlir::Location loc = getTokenLocation( offendingSymbol );

            if ( driverState.noVerboseParseError )
            {
                // coverage: syntax-error/array-return.silly
                emitUserError( loc, std::format( "parse error" ), currentFuncName );
            }
            else
            {
                // coverage: syntax-error/array-return-verbose.silly
                //
                // specifically implemented --no-verbose-parse-error to avoid error messages that change with any
                // grammar addition... but with the lit infra, it's easy enough to use wildcards and test this path
                // too.
                emitUserError( loc, std::format( "parse error: {}", msg ), currentFuncName );
            }
        }
        else
        {
            mlir::Location loc = builder.getUnknownLoc();
            emitInternalError( loc, __FILE__, __LINE__, __func__,
                               std::format( "parse error in {}:{}:{}: {}", sourceFile, line, charPositionInLine, msg ),
                               currentFuncName );
        }
    }

    mlir::Type Antlr4ParseListener::parseScalarType( const std::string &ty )
    {
        if ( ty == "BOOL" )
        {
            return typ.i1;
        }
        if ( ty == "INT8" )
        {
            return typ.i8;
        }
        if ( ty == "INT16" )
        {
            return typ.i16;
        }
        if ( ty == "INT32" )
        {
            return typ.i32;
        }
        if ( ty == "INT64" )
        {
            return typ.i64;
        }
        if ( ty == "FLOAT32" )
        {
            return typ.f32;
        }
        if ( ty == "FLOAT64" )
        {
            return typ.f64;
        }
        return nullptr;
    }

    void Antlr4ParseListener::enterStartRule( SillyParser::StartRuleContext *ctx )
    {
        assert( ctx );
        currentFuncName = ENTRY_SYMBOL_NAME;

        LocPairs locs = getLocations( ctx, true );

        if ( ctx->MODULE_TOKEN() )
        {
            isModule = true;
        }
        else
        {
            llvm::SmallVector<mlir::Location, 2> funcLocs{ locs.first, locs.second };
            mlir::Location fLoc = builder.getFusedLoc( funcLocs );

            createMain( fLoc, locs.first );
        }
    }

    void Antlr4ParseListener::enterImportStatement( SillyParser::ImportStatementContext *ctx )
    {
        mlir::Location loc = getStartLocation( ctx );
        assert( ctx );
        assert( ctx->IDENTIFIER() );
        std::string modname = ctx->IDENTIFIER()->getText();

        createImport( loc, modname );
    }

    void Antlr4ParseListener::enterScopedStatements( SillyParser::ScopedStatementsContext *ctx )
    {
        bool isFunctionBody = dynamic_cast<SillyParser::FunctionStatementContext *>( ctx->parent ) != nullptr;
        bool isForBody = dynamic_cast<SillyParser::ForStatementContext *>( ctx->parent ) != nullptr;

        mlir::Location loc = getStartLocation( ctx );

        createScopedStart( loc, !isFunctionBody and !isForBody );
    }

    void Antlr4ParseListener::exitScopedStatements( SillyParser::ScopedStatementsContext *ctx )
    {
        createScopedFinish();
    }

    mlir::Value Antlr4ParseListener::parseReturnExpression( mlir::Location loc,
                                                            SillyParser::ExpressionContext *expression,
                                                            LocationStack &ls )
    {
        mlir::Value value{};

        if ( expression )
        {
            mlir::Type returnType = lookupReturnType();

            if ( !returnType )
            {
                // coverage: syntax-error/return-expr-no-type.silly
                emitUserError( loc,
                               std::format( "return expression found '{}', but no return type for function {}",
                                            expression->getText(), currentFuncName ),
                               currentFuncName );
                return value;
            }

            value = parseExpression( expression, returnType, ls );
            if ( !value )
            {
                emitInternalError( loc, __FILE__, __LINE__, __func__, "parseExpression failed", currentFuncName );
                return value;
            }
        }

        return value;
    }

    void Antlr4ParseListener::exitStartRule( SillyParser::StartRuleContext *ctx )
    {
        assert( ctx );

        if ( !isModule and !ctx->exitStatement() )
        {
            LocPairs locs = getLocations( ctx );

            createMainExit( locs.second );
        }

#if 0
        LLVM_DEBUG( {
            llvm::errs() << "exitStartRule done: module dump:\n";
            rmod->dump();
        } );
#endif
    }

    void Antlr4ParseListener::enterFunctionStatement( SillyParser::FunctionStatementContext *ctx )
    {
        assert( ctx );
        LocPairs locs = getLocations( ctx, true );

        assert( ctx->IDENTIFIER() );
        std::string funcName = ctx->IDENTIFIER()->getText();

        mlir::Type returnType;
        if ( SillyParser::ScalarTypeContext *rt = ctx->scalarType() )
        {
            returnType = parseScalarType( rt->getText() );
            if ( !returnType )
            {
                mlir::Location loc = getStartLocation( ctx );
                emitInternalError( loc, __FILE__, __LINE__, __func__, "no returnType", currentFuncName );
                return;
            }
        }

        std::vector<mlir::Type> paramTypes;
        for ( SillyParser::VariableTypeAndNameContext *paramCtx : ctx->variableTypeAndName() )
        {
            assert( paramCtx->scalarType() );
            mlir::Type paramType = parseScalarType( paramCtx->scalarType()->getText() );
            paramTypes.push_back( paramType );
        }

        std::vector<std::string> paramNames;
        for ( SillyParser::VariableTypeAndNameContext *paramCtx : ctx->variableTypeAndName() )
        {
            assert( paramCtx->IDENTIFIER() );
            std::string paramName = paramCtx->IDENTIFIER()->getText();
            paramNames.push_back( paramName );
        }

        createFunctionStart( locs, funcName, !ctx->scopedStatements(), returnType, paramTypes, paramNames );
    }

    void Antlr4ParseListener::exitFunctionStatement( SillyParser::FunctionStatementContext *ctx )
    {
        assert( ctx );

        if ( SillyParser::ScopedStatementsContext *scope = ctx->scopedStatements() )
        {
            if ( !scope->returnStatement() )
            {
                mlir::Location loc = getStopLocation( scope );
                emitUserError( loc, "Function must have a RETURN statement", currentFuncName );
            }
        }

        createFunctionFinish();
    }

    mlir::Value Antlr4ParseListener::parseCallStatementOrExpr( SillyParser::CallExpressionContext *ctx,
                                                               bool callStatement, LocationStack &ls )
    {
        mlir::Value value{};
        assert( ctx );
        mlir::Location loc = getStartLocation( ctx );
        tNode *id = ctx->IDENTIFIER();
        std::string funcName = id->getText();
        ParserPerFunctionState &f = lookupFunctionState( funcName );
        mlir::func::FuncOp funcOp = f.getFuncOp();
        if ( !funcOp )
        {
            emitInternalError( loc, __FILE__, __LINE__, __func__, std::format( "null FuncOp for {}", funcName ),
                               currentFuncName );
            return value;
        }
        mlir::FunctionType funcType = funcOp.getFunctionType();

        std::vector<mlir::Value> parameters;

        if ( SillyParser::ParameterListContext *params = ctx->parameterList() )
        {
            int i = 0;

            assert( params );
            for ( SillyParser::ParameterExpressionContext *e : params->parameterExpression() )
            {
                SillyParser::ExpressionContext *p = e->expression();
                assert( p );
                std::string paramText = p->getText();
                // LLVM_DEBUG( { llvm::errs() << llvm::formatv( "CALL function {0}: param: {1}\n", funcName, paramText )
                // }
                // );

                mlir::Type ty = funcType.getInputs()[i];
                value = parseExpression( p, ty, ls );
                if ( !value )
                {
                    emitInternalError( loc, __FILE__, __LINE__, __func__, "parseExpression failed", currentFuncName );
                    return value;
                }

                parameters.push_back( value );
                i++;
            }
        }

        return createCall( loc, funcName, funcOp, funcType, callStatement, parameters, ls );
    }

    void Antlr4ParseListener::enterCallStatement( SillyParser::CallStatementContext *ctx )
    {
        assert( ctx );

        mlir::Location loc = getStartLocation( ctx );
        LocationStack ls( builder, loc );

        parseCallStatementOrExpr( ctx->callExpression(), true, ls );
    }

    void Antlr4ParseListener::enterDeclareHelper( mlir::Location loc, tNode *identifier, bool hasInitializer,
                                                  const std::vector<SillyParser::ExpressionContext *> &expressions,
                                                  SillyParser::ArrayBoundsExpressionContext *arrayBoundsExpression,
                                                  mlir::Type ty, LocationStack &ls )
    {
        if ( !identifier )
        {
            emitInternalError( loc, __FILE__, __LINE__, __func__, "no identifier for declaration processing",
                               currentFuncName );
            return;
        }
        std::string varName = identifier->getText();

        const std::vector<SillyParser::ExpressionContext *> *pExpressions{};

        if ( hasInitializer || expressions.size() )
        {
            pExpressions = &expressions;
        }

        std::string arrayBoundsString;
        mlir::Location aLoc = loc;
        if ( arrayBoundsExpression )
        {
            tNode *index = arrayBoundsExpression->INTEGER_PATTERN();
            arrayBoundsString = index->getText();
            aLoc = getTerminalLocation( index );
        }

        std::vector<mlir::Value> initializers;
        if ( pExpressions )
        {
            for ( SillyParser::ExpressionContext *e : *pExpressions )
            {
                mlir::Value init = parseExpression( e, ty, ls );
                if ( !init )
                {
                    emitInternalError( loc, __FILE__, __LINE__, __func__, "parseExpression failed", currentFuncName );
                    return;
                }

                initializers.push_back( init );
            }
        }

        createDeclaration( loc, varName, ty, aLoc, arrayBoundsString, pExpressions ? true : false, initializers, ls );
    }

    void Antlr4ParseListener::enterBoolDeclareStatement( SillyParser::BoolDeclareStatementContext *ctx )
    {
        assert( ctx );

        mlir::Location loc = getStartLocation( ctx );
        LocationStack ls( builder, loc );

        enterDeclareHelper( loc, ctx->IDENTIFIER(), ctx->EQUALS_TOKEN() || ctx->LEFT_CURLY_BRACKET_TOKEN(),
                            ctx->expression(), ctx->arrayBoundsExpression(), typ.i1, ls );
    }

    mlir::Type Antlr4ParseListener::integerDeclarationType( mlir::Location loc, SillyParser::IntTypeContext *ctx )
    {
        mlir::Type ty{};

        if ( ctx->INT8_TOKEN() )
        {
            ty = typ.i8;
        }
        else if ( ctx->INT16_TOKEN() )
        {
            ty = typ.i16;
        }
        else if ( ctx->INT32_TOKEN() )
        {
            ty = typ.i32;
        }
        else if ( ctx->INT64_TOKEN() )
        {
            ty = typ.i64;
        }
        else
        {
            emitInternalError( loc, __FILE__, __LINE__, __func__, "Unsupported signed integer declaration size.",
                               currentFuncName );
        }

        return ty;
    }

    void Antlr4ParseListener::enterIntDeclareStatement( SillyParser::IntDeclareStatementContext *ctx )
    {
        assert( ctx );
        mlir::Location loc = getStartLocation( ctx );
        LocationStack ls( builder, loc );

        mlir::Type ty = integerDeclarationType( loc, ctx->intType() );

        enterDeclareHelper( loc, ctx->IDENTIFIER(), ctx->EQUALS_TOKEN() || ctx->LEFT_CURLY_BRACKET_TOKEN(),
                            ctx->expression(), ctx->arrayBoundsExpression(), ty, ls );
    }

    void Antlr4ParseListener::enterFloatDeclareStatement( SillyParser::FloatDeclareStatementContext *ctx )
    {
        assert( ctx );
        mlir::Type ty;

        mlir::Location loc = getStartLocation( ctx );
        LocationStack ls( builder, loc );

        if ( ctx->FLOAT32_TOKEN() )
        {
            ty = typ.f32;
        }
        else if ( ctx->FLOAT64_TOKEN() )
        {
            ty = typ.f64;
        }
        else
        {
            mlir::Location loc = getStartLocation( ctx );
            emitInternalError( loc, __FILE__, __LINE__, __func__, "Unsupported floating point declaration size.",
                               currentFuncName );
            return;
        }

        enterDeclareHelper( loc, ctx->IDENTIFIER(), ctx->EQUALS_TOKEN() || ctx->LEFT_CURLY_BRACKET_TOKEN(),
                            ctx->expression(), ctx->arrayBoundsExpression(), ty, ls );
    }

    void Antlr4ParseListener::enterStringDeclareStatement( SillyParser::StringDeclareStatementContext *ctx )
    {
        assert( ctx );
        mlir::Location loc = getStartLocation( ctx );
        LocationStack ls( builder, loc );

        assert( ctx->IDENTIFIER() );
        std::string varName = ctx->IDENTIFIER()->getText();
        SillyParser::ArrayBoundsExpressionContext *arrayBounds = ctx->arrayBoundsExpression();

        tNode *index = arrayBounds->INTEGER_PATTERN();
        const std::string arrayBoundsString = index->getText();
        mlir::Location aloc = getTerminalLocation( index );
        std::vector<mlir::Value> initializers;
        createDeclaration( loc, varName, typ.i8, aloc, arrayBoundsString, false, initializers, ls );

        if ( tNode *theString = ctx->STRING_PATTERN() )
        {
            silly::DeclareOp declareOp = lookupDeclareForVar( loc, varName );
            mlir::Value var = declareOp.getResult();

            silly::StringLiteralOp stringLiteral = createStringLiteral( loc, theString->getText(), ls );
            if ( stringLiteral )
            {
                mlir::Value i{};

                // mlir::Location fusedLoc = ls.fuseLocations( );
                silly::AssignOp::create( builder, loc, var, i, stringLiteral );
            }
        }
    }

    void Antlr4ParseListener::checkForReturnInScope( SillyParser::ScopedStatementsContext *scope, const char *what )
    {
        assert( scope );

        if ( SillyParser::ReturnStatementContext *ret = scope->returnStatement() )
        {
            mlir::Location rLoc = getStartLocation( ret );
            emitUserError( rLoc, std::format( "RETURN is not currently allowed in a {}", what ), currentFuncName );
        }
    }

    void Antlr4ParseListener::enterIfStatement( SillyParser::IfStatementContext *ctx )
    {
        assert( ctx );

        mlir::Location loc = getStartLocation( ctx );
        LocationStack ls( builder, loc );

        checkForReturnInScope( ctx->scopedStatements(), "IF block" );

        mlir::Value conditionPredicate = parseExpression( ctx->expression(), {}, ls );
        if ( !conditionPredicate )
        {
            emitInternalError( loc, __FILE__, __LINE__, __func__, "parseExpression failed", currentFuncName );
            return;
        }

        createIf( loc, conditionPredicate, true, ls );
    }

    void Antlr4ParseListener::enterElseStatement( SillyParser::ElseStatementContext *ctx )
    {
        assert( ctx );
        mlir::Location loc = getStartLocation( ctx );

        checkForReturnInScope( ctx->scopedStatements(), "ELSE block" );

        createElseBlockSelection( loc );
    }

    void Antlr4ParseListener::enterElifStatement( SillyParser::ElifStatementContext *ctx )
    {
        assert( ctx );
        mlir::Location loc = getStartLocation( ctx );
        LocationStack ls( builder, loc );

        checkForReturnInScope( ctx->scopedStatements(), "ELIF block" );

        createElseBlockSelection( loc );

        mlir::Value conditionPredicate = parseExpression( ctx->expression(), {}, ls );
        if ( !conditionPredicate )
        {
            emitInternalError( loc, __FILE__, __LINE__, __func__, "parseExpression failed", currentFuncName );
            return;
        }

        createIf( loc, conditionPredicate, false, ls );
    }

    void Antlr4ParseListener::exitIfElifElseStatement( SillyParser::IfElifElseStatementContext *ctx )
    {
        createIfElifElseFinish();
    }

    void Antlr4ParseListener::enterForStatement( SillyParser::ForStatementContext *ctx )
    {
        assert( ctx );
        mlir::Location loc = getStartLocation( ctx );
        LocationStack ls( builder, loc );

        LLVM_DEBUG( { llvm::errs() << llvm::formatv( "For: {0}\n", ctx->getText() ); } );

        assert( ctx->IDENTIFIER() );
        std::string varName = ctx->IDENTIFIER()->getText();

        assert( ctx->forStart() );
        assert( ctx->forEnd() );
        assert( ctx->forStart()->forRangeExpression() );
        assert( ctx->forEnd()->forRangeExpression() );
        SillyParser::ExpressionContext *pStart = ctx->forStart()->forRangeExpression()->expression();
        SillyParser::ExpressionContext *pEnd = ctx->forEnd()->forRangeExpression()->expression();
        SillyParser::ExpressionContext *pStep{};
        if ( SillyParser::ForStepContext *st = ctx->forStep() )
        {
            assert( st->forRangeExpression() );
            pStep = st->forRangeExpression()->expression();
        }

        mlir::Value start;
        mlir::Value end;
        mlir::Value step;

        mlir::Type elemType = integerDeclarationType( loc, ctx->intType() );

        std::string s;
        if ( pStart )
        {
            start = parseExpression( pStart, elemType, ls );
            if ( !start )
            {
                emitInternalError( loc, __FILE__, __LINE__, __func__, "parseExpression failed", currentFuncName );
                return;
            }
        }
        else
        {
            emitInternalError( loc, __FILE__, __LINE__, __func__,
                               std::format( "FOR loop: expected start index: {}", ctx->getText() ), currentFuncName );
            return;
        }

        if ( pEnd )
        {
            end = parseExpression( pEnd, elemType, ls );
            if ( !end )
            {
                emitInternalError( loc, __FILE__, __LINE__, __func__, "parseExpression failed", currentFuncName );
                return;
            }
        }
        else
        {
            emitInternalError( loc, __FILE__, __LINE__, __func__,
                               std::format( "FOR loop: expected end index: {}", ctx->getText() ), currentFuncName );
            return;
        }

        if ( pStep )
        {
            step = parseExpression( pStep, elemType, ls );
            if ( !step )
            {
                emitInternalError( loc, __FILE__, __LINE__, __func__, "parseExpression failed", currentFuncName );
                return;
            }
        }

        checkForReturnInScope( ctx->scopedStatements(), "FOR loop body" );

        mlir::Location varLoc = getTerminalLocation( ctx->IDENTIFIER() );

        createForStart( loc, varName, elemType, varLoc, start, end, step, ls );
    }

    void Antlr4ParseListener::exitForStatement( SillyParser::ForStatementContext *ctx )
    {
        assert( ctx );
        mlir::Location loc = getStartLocation( ctx );

        createForFinish( loc );
    }

    void Antlr4ParseListener::handlePrint( mlir::Location loc,
                                           const std::vector<SillyParser::ExpressionContext *> &args,
                                           const std::string &errorContextString, PrintFlags pf, LocationStack &ls )
    {
        std::vector<mlir::Value> vargs;
        for ( SillyParser::ExpressionContext *parg : args )
        {
            mlir::Value v = parseExpression( parg, {}, ls );
            if ( !v )
            {
                emitInternalError( loc, __FILE__, __LINE__, __func__, "parseExpression failed", currentFuncName );
                return;
            }

            vargs.push_back( v );
        }

        ls.push_back( loc );
        mlir::arith::ConstantIntOp constFlagOp = mlir::arith::ConstantIntOp::create( builder, loc, pf, 32 );

        // mlir::Location fusedLoc = ls.fuseLocations( );
        silly::PrintOp::create( builder, loc, constFlagOp, vargs );
    }

    void Antlr4ParseListener::enterPrintStatement( SillyParser::PrintStatementContext *ctx )
    {
        assert( ctx );
        mlir::Location loc = getStartLocation( ctx );
        LocationStack ls( builder, loc );

        int flags = PRINT_FLAGS_NONE;
        if ( ctx->CONTINUE_TOKEN() )
        {
            flags = PRINT_FLAGS_CONTINUE;
        }
        handlePrint( loc, ctx->expression(), ctx->getText(), (PrintFlags)flags, ls );
    }

    void Antlr4ParseListener::enterErrorStatement( SillyParser::ErrorStatementContext *ctx )
    {
        assert( ctx );
        mlir::Location loc = getStartLocation( ctx );
        LocationStack ls( builder, loc );

        int flags = PRINT_FLAGS_ERROR;
        if ( ctx->CONTINUE_TOKEN() )
        {
            flags |= PRINT_FLAGS_CONTINUE;
        }
        handlePrint( loc, ctx->expression(), ctx->getText(), (PrintFlags)flags, ls );
    }

    void Antlr4ParseListener::enterAbortStatement( SillyParser::AbortStatementContext *ctx )
    {
        assert( ctx );
        mlir::Location loc = getStartLocation( ctx );

        silly::AbortOp::create( builder, loc );
    }

    void Antlr4ParseListener::enterGetStatement( SillyParser::GetStatementContext *ctx )
    {
        assert( ctx );
        mlir::Location loc = getStartLocation( ctx );
        LocationStack ls( builder, loc );

        std::string varName;
        mlir::Value indexValue;
        mlir::Location iloc = loc;
        SillyParser::ScalarOrArrayElementContext *scalarOrArrayElement = ctx->scalarOrArrayElement();

        if ( scalarOrArrayElement )
        {
            tNode *varNameObject = scalarOrArrayElement->IDENTIFIER();
            assert( varNameObject );
            varName = varNameObject->getText();

            if ( SillyParser::IndexExpressionContext *indexExpr = scalarOrArrayElement->indexExpression() )
            {
                iloc = getStartLocation( indexExpr->expression() );
                indexValue = parseExpression( indexExpr->expression(), {}, ls );
                if ( !indexValue )
                {
                    emitInternalError( loc, __FILE__, __LINE__, __func__, "parseExpression failed", currentFuncName );
                    return;
                }
            }
        }
        else
        {
            emitInternalError( loc, __FILE__, __LINE__, __func__,
                               std::format( "unexpected get context {}", ctx->getText() ), currentFuncName );
        }

        createGet( loc, varName, indexValue, iloc, ls );
    }

    void Antlr4ParseListener::enterReturnStatement( SillyParser::ReturnStatementContext *ctx )
    {
        assert( ctx );
        LocPairs locs = getLocations( ctx );
        LocationStack ls( builder, locs.first );

        mlir::Value value = parseReturnExpression( locs.second, ctx->expression(), ls );
        createReturnLike( locs.second, value, ls );
    }

    void Antlr4ParseListener::enterExitStatement( SillyParser::ExitStatementContext *ctx )
    {
        assert( ctx );
        LocPairs locs = getLocations( ctx );
        LocationStack ls( builder, locs.first );

        mlir::Value value = parseReturnExpression( locs.second, ctx->expression(), ls );
        createReturnLike( locs.second, value, ls );
    }

    void Antlr4ParseListener::enterAssignmentStatement( SillyParser::AssignmentStatementContext *ctx )
    {
        assert( ctx );
        mlir::Location loc = getStartLocation( ctx );
        LocationStack ls( builder, loc );

        SillyParser::ScalarOrArrayElementContext *lhs = ctx->scalarOrArrayElement();
        assert( lhs );
        assert( lhs->IDENTIFIER() );
        std::string currentVarName = lhs->IDENTIFIER()->getText();

        SillyParser::IndexExpressionContext *indexExpr = lhs->indexExpression();
        mlir::Value currentIndexExpr = mlir::Value{};

        bool declared = lookupVariableDeclaration( currentVarName );
        if ( !declared )
        {
            // coverage: syntax-error/undeclared-var.silly
            emitUserError( loc, std::format( "Attempt to assign to undeclared variable: {}", currentVarName ),
                           currentFuncName );
            return;
        }

        if ( indexExpr )
        {
            currentIndexExpr = parseExpression( indexExpr->expression(), {}, ls );
            if ( !currentIndexExpr )
            {
                emitInternalError( loc, __FILE__, __LINE__, __func__, "parseExpression failed", currentFuncName );
                return;
            }
        }

        SillyParser::ExpressionContext *exprContext = ctx->expression();
        mlir::Location aLoc = getStartLocation( exprContext );
        mlir::Value resultValue = parseExpression( exprContext, {}, ls );
        createAssignment( aLoc, resultValue, currentVarName, currentIndexExpr, ls );
    }

    mlir::Value Antlr4ParseListener::parseOr( antlr4::ParserRuleContext *ctx, mlir::Type ty, LocationStack &ls )
    {
        assert( ctx );

        if ( SillyParser::OrExprContext *orCtx = dynamic_cast<SillyParser::OrExprContext *>( ctx ) )
        {
            mlir::Location loc = getStartLocation( ctx );

            // We have at least one OR
            std::vector<SillyParser::BinaryExpressionOrContext *> orOperands = orCtx->binaryExpressionOr();

            // First operand (no special case needed)
            mlir::Value value = parseXor( orOperands[0], ty, ls );
            if ( !value )
            {
                emitInternalError( loc, __FILE__, __LINE__, __func__, "parseXor failed", currentFuncName );
                return value;
            }

            for ( size_t i = 1; i < orOperands.size(); ++i )
            {
                mlir::Value rhs = parseXor( orOperands[i], ty, ls );
                if ( !rhs )
                {
                    emitInternalError( loc, __FILE__, __LINE__, __func__, "parseXor failed", currentFuncName );
                    return rhs;
                }

                mlir::Type ty = biggestTypeOf( value.getType(), rhs.getType() );

                value = createBinaryArith( loc, silly::ArithBinOpKind::Or, ty, value, rhs, ls );
            }

            return value;
        }

        // No OR: just descend to the next level (XOR / single-term)
        return parseXor( ctx, ty, ls );
    }

    mlir::Value Antlr4ParseListener::parseXor( antlr4::ParserRuleContext *ctx, mlir::Type ty, LocationStack &ls )
    {
        assert( ctx );

        if ( SillyParser::XorExprContext *xorCtx = dynamic_cast<SillyParser::XorExprContext *>( ctx ) )
        {
            mlir::Location loc = getStartLocation( ctx );

            // Has XOR operator(s)
            std::vector<SillyParser::BinaryExpressionXorContext *> xorOperands = xorCtx->binaryExpressionXor();

            mlir::Value value = parseAnd( xorOperands[0], ty, ls );
            if ( !value )
            {
                emitInternalError( loc, __FILE__, __LINE__, __func__, "parseAnd failed", currentFuncName );
                return value;
            }

            for ( size_t i = 1; i < xorOperands.size(); ++i )
            {
                mlir::Value rhs = parseAnd( xorOperands[i], ty, ls );
                if ( !rhs )
                {
                    emitInternalError( loc, __FILE__, __LINE__, __func__, "parseAnd failed", currentFuncName );
                    return rhs;
                }

                mlir::Type ty = biggestTypeOf( value.getType(), rhs.getType() );

                value = createBinaryArith( loc, silly::ArithBinOpKind::Xor, ty, value, rhs, ls );
            }

            return value;
        }

        // No XOR: descend to AND level
        return parseAnd( ctx, ty, ls );
    }

    mlir::Value Antlr4ParseListener::parseAnd( antlr4::ParserRuleContext *ctx, mlir::Type ty, LocationStack &ls )
    {
        assert( ctx );

        // Check whether this context actually contains AND operators
        SillyParser::AndExprContext *andCtx = dynamic_cast<SillyParser::AndExprContext *>( ctx );

        if ( andCtx )
        {
            mlir::Location loc = getStartLocation( ctx );

            // We have one or more AND operators
            std::vector<SillyParser::BinaryExpressionAndContext *> andOperands = andCtx->binaryExpressionAnd();

            // First operand
            mlir::Value value = parseEquality( andOperands[0], ty, ls );
            if ( !value )
            {
                emitInternalError( loc, __FILE__, __LINE__, __func__, "parseEquality failed", currentFuncName );
                return value;
            }

            // Fold the remaining ANDs (left associative)
            for ( size_t i = 1; i < andOperands.size(); ++i )
            {
                mlir::Value rhs = parseEquality( andOperands[i], ty, ls );
                if ( !rhs )
                {
                    emitInternalError( loc, __FILE__, __LINE__, __func__, "parseEquality failed", currentFuncName );
                    return rhs;
                }

                mlir::Type ty = biggestTypeOf( value.getType(), rhs.getType() );

                value = createBinaryArith( loc, silly::ArithBinOpKind::And, ty, value, rhs, ls );
            }

            return value;
        }

        // No AND operator: descend directly to the next level (equality)
        return parseEquality( ctx, ty, ls );
    }

    mlir::Value Antlr4ParseListener::parseEquality( antlr4::ParserRuleContext *ctx, mlir::Type ty, LocationStack &ls )
    {
        assert( ctx );
        mlir::Value value{};

        // Check if this is the concrete alternative that has EQ / NE operators
        if ( SillyParser::EqNeExprContext *eqNeCtx = dynamic_cast<SillyParser::EqNeExprContext *>( ctx ) )
        {
            mlir::Location loc = getStartLocation( ctx );

            // We have an EQ / NE operator
            std::vector<SillyParser::BinaryExpressionCompareContext *> operands = eqNeCtx->binaryExpressionCompare();

            // First (leftmost) operand
            value = parseComparison( operands[0], ty, ls );
            if ( !value )
            {
                emitInternalError( loc, __FILE__, __LINE__, __func__, "parseComparison failed", currentFuncName );
                return value;
            }

            assert( operands.size() <= 2 );

            if ( operands.size() == 2 )
            {
                mlir::Value rhs = parseComparison( operands[1], ty, ls );
                if ( !rhs )
                {
                    emitInternalError( loc, __FILE__, __LINE__, __func__, "parseComparison failed", currentFuncName );
                    return rhs;
                }

                if ( eqNeCtx->equalityOperator()->EQUALITY_TOKEN() )
                {
                    value = createBinaryCompare( loc, silly::CmpBinOpKind::Equal, value, rhs, ls );
                }
                else if ( eqNeCtx->equalityOperator()->NOTEQUAL_TOKEN() )
                {
                    value = createBinaryCompare( loc, silly::CmpBinOpKind::NotEqual, value, rhs, ls );
                }
                else
                {
                    emitInternalError( loc, __FILE__, __LINE__, __func__,
                                       std::format( "missing EQ or NE token: {}", ctx->getText() ), currentFuncName );
                    value = mlir::Value{};
                }
            }

            return value;
        }

        // No EQ or NE operators: descend directly to comparison level
        return parseComparison( ctx, ty, ls );
    }

    mlir::Value Antlr4ParseListener::parseComparison( antlr4::ParserRuleContext *ctx, mlir::Type ty, LocationStack &ls )
    {
        assert( ctx );
        mlir::Value value{};

        // Check if this context actually contains comparison operators
        // (the # compareExpr alternative has the repetition)
        if ( SillyParser::CompareExprContext *compareCtx = dynamic_cast<SillyParser::CompareExprContext *>( ctx ) )
        {
            mlir::Location loc = getStartLocation( ctx );

            // We have a comparison operator
            std::vector<SillyParser::BinaryExpressionAddSubContext *> operands = compareCtx->binaryExpressionAddSub();

            assert( operands.size() <= 2 );

            // First (leftmost) operand
            value = parseAdditive( operands[0], ty, ls );
            if ( !value )
            {
                emitInternalError( loc, __FILE__, __LINE__, __func__, "parseAdditive failed", currentFuncName );
                return value;
            }

            if ( operands.size() == 2 )
            {
                mlir::Value rhs = parseAdditive( operands[1], ty, ls );
                if ( !rhs )
                {
                    emitInternalError( loc, __FILE__, __LINE__, __func__, "parseAdditive failed", currentFuncName );
                    return rhs;
                }

                SillyParser::RelationalOperatorContext *op = compareCtx->relationalOperator();
                assert( op );

                if ( op->LESSTHAN_TOKEN() )
                {
                    value = createBinaryCompare( loc, silly::CmpBinOpKind::Less, value, rhs, ls );
                }
                else if ( op->GREATERTHAN_TOKEN() )
                {
                    value = createBinaryCompare( loc, silly::CmpBinOpKind::Less, rhs, value, ls );
                }
                else if ( op->LESSEQUAL_TOKEN() )
                {
                    value = createBinaryCompare( loc, silly::CmpBinOpKind::LessEq, value, rhs, ls );
                }
                else if ( op->GREATEREQUAL_TOKEN() )
                {
                    value = createBinaryCompare( loc, silly::CmpBinOpKind::LessEq, rhs, value, ls );
                }
                else
                {
                    emitInternalError( loc, __FILE__, __LINE__, __func__,
                                       std::format( "missing comparison operator: {}", ctx->getText() ),
                                       currentFuncName );
                    value = mlir::Value{};
                }
            }

            return value;
        }

        // No comparison operators : descend directly to additive level
        return parseAdditive( ctx, ty, ls );
    }

    mlir::Value Antlr4ParseListener::parseAdditive( antlr4::ParserRuleContext *ctx, mlir::Type ty, LocationStack &ls )
    {
        assert( ctx );
        mlir::Value value{};

        // Check if this context actually contains + or - operators
        // (AddSubExprContext is the alternative that has the repetition)
        if ( SillyParser::AddSubExprContext *addSubCtx = dynamic_cast<SillyParser::AddSubExprContext *>( ctx ) )
        {
            mlir::Location loc = getStartLocation( ctx );

            std::vector<SillyParser::AdditionOperatorContext *> ops = addSubCtx->additionOperator();

            // We have one or more + or - operators
            std::vector<SillyParser::BinaryExpressionMulDivContext *> operands = addSubCtx->binaryExpressionMulDiv();
            size_t numOperands = operands.size();
            assert( ( ops.size() + 1 ) == numOperands );

            // First operand
            value = parseMultiplicative( operands[0], ty, ls );
            if ( !value )
            {
                emitInternalError( loc, __FILE__, __LINE__, __func__, "parseMultiplicative failed", currentFuncName );
                return value;
            }

            // Fold the remaining additions/subtractions left-associatively
            for ( size_t i = 1; i < numOperands; ++i )
            {
                mlir::Value rhs = parseMultiplicative( operands[i], ty, ls );
                if ( !rhs )
                {
                    emitInternalError( loc, __FILE__, __LINE__, __func__, "parseMultiplicative failed",
                                       currentFuncName );
                    return rhs;
                }

                mlir::Type ty = biggestTypeOf( value.getType(), rhs.getType() );

                SillyParser::AdditionOperatorContext *op = ops[i - 1];
                if ( op->PLUSCHAR_TOKEN() )
                {
                    value = createBinaryArith( loc, silly::ArithBinOpKind::Add, ty, value, rhs, ls );
                }
                else if ( op->MINUS_TOKEN() )
                {
                    value = createBinaryArith( loc, silly::ArithBinOpKind::Sub, ty, value, rhs, ls );
                }
                else
                {
                    emitInternalError( loc, __FILE__, __LINE__, __func__,
                                       std::format( "missing + or - operator at index {}: {}", i - 1, ctx->getText() ),
                                       currentFuncName );
                    break;
                }
            }

            return value;
        }
        // Descend directly to multiplicative level if no +-
        return parseMultiplicative( ctx, ty, ls );
    }

    mlir::Value Antlr4ParseListener::parseMultiplicative( antlr4::ParserRuleContext *ctx, mlir::Type ty,
                                                          LocationStack &ls )
    {
        assert( ctx );
        mlir::Value value{};

        // Check whether this context actually contains * or / operators
        // (MulDivExprContext is the alternative that has the repetition)
        if ( SillyParser::MulDivExprContext *mulDivCtx = dynamic_cast<SillyParser::MulDivExprContext *>( ctx ) )
        {
            mlir::Location loc = getStartLocation( ctx );

            std::vector<SillyParser::MultiplicativeOperatorContext *> ops = mulDivCtx->multiplicativeOperator();

            // We have one or more * or / operators
            std::vector<SillyParser::UnaryExpressionContext *> operands = mulDivCtx->unaryExpression();
            size_t numOperands = operands.size();
            assert( ( ops.size() + 1 ) == numOperands );

            // First operand
            value = parseUnary( operands[0], ty, ls );
            if ( !value )
            {
                emitInternalError( loc, __FILE__, __LINE__, __func__, "parseUnary failed", currentFuncName );
                return value;
            }

            // Fold the remaining multiplications/divisions left-associatively
            for ( size_t i = 1; i < numOperands; ++i )
            {
                mlir::Value rhs = parseUnary( operands[i], ty, ls );
                if ( !rhs )
                {
                    emitInternalError( loc, __FILE__, __LINE__, __func__, "parseUnary failed", currentFuncName );
                    return rhs;
                }

                mlir::Type ty = biggestTypeOf( value.getType(), rhs.getType() );

                SillyParser::MultiplicativeOperatorContext *op = ops[i - 1];

                if ( op->TIMES_TOKEN() )
                {
                    value = createBinaryArith( loc, silly::ArithBinOpKind::Mul, ty, value, rhs, ls );
                }
                else if ( op->DIV_TOKEN() )
                {
                    value = createBinaryArith( loc, silly::ArithBinOpKind::Div, ty, value, rhs, ls );
                }
                else if ( op->MOD_TOKEN() )
                {
                    value = createBinaryArith( loc, silly::ArithBinOpKind::Mod, ty, value, rhs, ls );
                }
                else
                {
                    emitInternalError( loc, __FILE__, __LINE__, __func__,
                                       std::format( "missing * or / operator at index {}", i - 1 ), currentFuncName );
                    break;
                }
            }

            return value;
        }

        // Descend directly to unary level if no * or /
        return parseUnary( ctx, ty, ls );
    }

    mlir::Value Antlr4ParseListener::parseUnary( antlr4::ParserRuleContext *ctx, mlir::Type ty, LocationStack &ls )
    {
        assert( ctx );
        mlir::Location loc = getStartLocation( ctx );
        mlir::Value value{};

        if ( SillyParser::UnaryOpContext *unaryOpCtx = dynamic_cast<SillyParser::UnaryOpContext *>( ctx ) )
        {
            // Case 1: unary operator applied to another unary expression (# unaryOp)
            SillyParser::UnaryOperatorContext *unaryOp = unaryOpCtx->unaryOperator();
            assert( unaryOp );

            // Recurse to the inner unary expression
            value = parseUnary( unaryOpCtx->unaryExpression(), ty, ls );
            if ( !value )
            {
                emitInternalError( loc, __FILE__, __LINE__, __func__, "parseUnary failed", currentFuncName );
                return value;
            }

            std::string opText = unaryOp->getText();

            ls.push_back( loc );

            UnaryOp op{ UnaryOp::Plus };
            if ( opText == "-" )
            {
                op = UnaryOp::Negate;
            }
            else if ( opText == "NOT" )
            {
                op = UnaryOp::Not;
            }
            value = createUnary( loc, value, op, ls );
        }
        else if ( SillyParser::PrimaryContext *primaryCtx = dynamic_cast<SillyParser::PrimaryContext *>( ctx ) )
        {
            // Case 2: no unary operator, just a primary (# primary)
            value = parsePrimary( primaryCtx->primaryExpression(), ty, ls );
        }
        else
        {
            emitInternalError( loc, __FILE__, __LINE__, __func__,
                               std::format( "unknown unary context: {}", ctx->getText() ), currentFuncName );
            return value;
        }

#if 0
        LLVM_DEBUG( {
            llvm::errs() << "parseUnary: " << ctx->getText() << " -> type ";
            value.getType().dump();
            llvm::errs() << "\n";
        } );
#endif

        return value;
    }

    mlir::Value Antlr4ParseListener::parsePrimary( antlr4::ParserRuleContext *ctx, mlir::Type ty, LocationStack &ls )
    {
        assert( ctx );
        mlir::Location loc = getStartLocation( ctx );
        mlir::Value value{};

        if ( SillyParser::LitPrimaryContext *litCtx = dynamic_cast<SillyParser::LitPrimaryContext *>( ctx ) )
        {
            // Literal case (# litPrimary)
            SillyParser::LiteralContext *lit = litCtx->literal();
            assert( lit );

            if ( tNode *booleanNode = lit->BOOLEAN_PATTERN() )
            {
                value = createBooleanFromString( loc, booleanNode->getText(), ls );
            }
            else if ( tNode *integerNode = lit->INTEGER_PATTERN() )
            {
                unsigned width = 64;

                if ( ty )
                {
                    if ( mlir::IntegerType ity = mlir::dyn_cast<mlir::IntegerType>( ty ) )
                    {
                        width = ity.getWidth();
                    }
                }

                value = createIntegerFromString( loc, width, integerNode->getText(), ls );
            }
            else if ( tNode *floatNode = lit->FLOAT_PATTERN() )
            {
                mlir::FloatType fty{};
                if ( ty )
                {
                    fty = mlir::dyn_cast<mlir::FloatType>( ty );
                }

                if ( !fty )
                {
                    fty = typ.f64;
                }

                value = createFloatFromString( loc, fty, floatNode->getText(), ls );
            }
            else if ( tNode *stringNode = lit->STRING_PATTERN() )
            {
                silly::StringLiteralOp stringLiteral = createStringLiteral( loc, stringNode->getText(), ls );

                if ( stringLiteral )
                {
                    value = stringLiteral.getResult();
                }
                else
                {
                    return value;
                }
            }
            else
            {
                emitInternalError( loc, __FILE__, __LINE__, __func__,
                                   std::format( "unknown literal type in primary: {}", ctx->getText() ),
                                   currentFuncName );
                return value;
            }
        }
        else if ( SillyParser::VarPrimaryContext *varCtx = dynamic_cast<SillyParser::VarPrimaryContext *>( ctx ) )
        {
            // Variable / array element (# varPrimary) / induction-variable
            SillyParser::ScalarOrArrayElementContext *scalarOrArrayElement = varCtx->scalarOrArrayElement();
            assert( scalarOrArrayElement );

            tNode *variableNode = scalarOrArrayElement->IDENTIFIER();
            assert( variableNode );
            std::string varName = variableNode->getText();

            mlir::Value iValue;
            mlir::Location iLoc = loc;

            if ( SillyParser::IndexExpressionContext *indexExpr = scalarOrArrayElement->indexExpression() )
            {
                iValue = parseExpression( indexExpr->expression(), {}, ls );
                if ( !iValue )
                {
                    emitInternalError( loc, __FILE__, __LINE__, __func__, "parseExpression failed", currentFuncName );
                    return value;
                }

                iLoc = getStartLocation( indexExpr->expression() );
            }

            value = createVariableLoadOrLookup( loc, varName, iValue, iLoc, ls );
        }
        else if ( SillyParser::CallPrimaryContext *callCtx = dynamic_cast<SillyParser::CallPrimaryContext *>( ctx ) )
        {
            // Function call (# callPrimary)
            value = parseCallStatementOrExpr( callCtx->callExpression(), false, ls );
        }
        else if ( SillyParser::ParenExprContext *parenCtx = dynamic_cast<SillyParser::ParenExprContext *>( ctx ) )
        {
            // Parenthesized expression (# parenExpr)
            value = parseExpression( parenCtx->expression(), {}, ls );
            if ( !value )
            {
                emitInternalError( loc, __FILE__, __LINE__, __func__, "parseExpression failed", currentFuncName );
                return value;
            }
        }
        else
        {
            emitInternalError( loc, __FILE__, __LINE__, __func__,
                               std::format( "unknown primary expression: {}", ctx->getText() ), currentFuncName );
            return value;
        }

        if ( !value )
        {
            emitInternalError( loc, __FILE__, __LINE__, __func__,
                               std::format( "expression parse error: {}", ctx->getText() ), currentFuncName );
            return value;
        }

#if 0
        LLVM_DEBUG( {
            llvm::errs() << "parsePrimary: " << ctx->getText() << " -> type ";
            value.getType().dump();
            llvm::errs() << "\n";
        } );
#endif

        return value;
    }

    mlir::OwningOpRef<mlir::ModuleOp> runParseListener( silly::SourceManager &s, const std::string &filename )
    {
        Antlr4ParseListener listener( s, filename );

        return listener.run();
    }
}    // namespace silly

// vim: et ts=4 sw=4
