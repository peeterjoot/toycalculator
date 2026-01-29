///
/// @file    parser.cpp
/// @author  Peeter Joot <peeterjoot@pm.me>
/// @brief   altlr4 parse tree listener and MLIR builder.
///
#include <llvm/Support/Debug.h>
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

#include "SillyExceptions.hpp"
#include "parser.hpp"

#define DEBUG_TYPE "silly-parser"

#define ENTRY_SYMBOL_NAME "main"

#define CATCH_USER_ERROR                                \
    catch ( const UserError &e )                        \
    {                                                   \
        mlir::emitError( e.getLocation() ) << e.what(); \
        hasErrors = true;                               \
    }

namespace silly
{
    inline std::string mlirTypeToString( mlir::Type t )
    {
        std::string s;
        llvm::raw_string_ostream( s ) << t;
        return s;
    }

    inline PerFunctionState &ParseListener::funcState( const std::string &funcName )
    {
        if ( !functionStateMap.contains( funcName ) )
        {
            functionStateMap[funcName] = std::make_unique<PerFunctionState>();
        }

        return *functionStateMap[funcName];
    }

    inline mlir::Value ParseListener::searchForInduction( const std::string &varName )
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

    inline void ParseListener::pushInductionVariable( const std::string &varName, mlir::Value i )
    {
        inductionVariables.emplace_back( varName, i );
    }

    inline void ParseListener::popInductionVariable()
    {
        inductionVariables.pop_back();
    }

    inline std::string formatLocation( mlir::Location loc )
    {
        if ( mlir::FileLineColLoc fileLoc = mlir::dyn_cast<mlir::FileLineColLoc>( loc ) )
        {
            return std::format( "{}:{}:{}: ", fileLoc.getFilename().str(), fileLoc.getLine(), fileLoc.getColumn() );
        }
        return "";
    }

    inline mlir::Value ParseListener::parseBoolean( mlir::Location loc, const std::string &s )
    {
        int val;
        if ( s == "TRUE" )
        {
            val = 1;
        }
        else if ( s == "FALSE" )
        {
            val = 0;
        }
        else
        {
            throw ExceptionWithContext( __FILE__, __LINE__, __func__,
                                        std::format( "{}error: internal error: boolean value neither TRUE nor "
                                                     "FALSE: {}.\n",
                                                     formatLocation( loc ), s ) );
        }

        return builder.create<mlir::arith::ConstantIntOp>( loc, val, 1 );
    }

    inline mlir::Value ParseListener::parseInteger( mlir::Location loc, int width, const std::string &s )
    {
        int64_t val = std::stoll( s );

        return builder.create<mlir::arith::ConstantIntOp>( loc, val, width );
    }

    inline mlir::Value ParseListener::parseFloat( mlir::Location loc, mlir::FloatType ty, const std::string &s )
    {
        if ( ty == tyF32 )
        {
            float val = std::stof( s );

            llvm::APFloat apVal( val );

            return builder.create<mlir::arith::ConstantFloatOp>( loc, tyF32, apVal );
        }
        else
        {
            double val = std::stod( s );

            llvm::APFloat apVal( val );

            return builder.create<mlir::arith::ConstantFloatOp>( loc, tyF64, apVal );
        }
    }

    inline std::string ParseListener::stripQuotes( mlir::Location loc, const std::string &input ) const
    {
        if ( ( input.size() < 2 ) || ( input.front() != '"' ) || ( input.back() != '"' ) )
        {
            throw ExceptionWithContext(
                __FILE__, __LINE__, __func__,
                std::format( "{}internal error: String '{}' was not double quotes enclosed as expected.\n",
                             formatLocation( loc ), input ) );
        }

        return input.substr( 1, input.size() - 2 );
    }

    inline LocPairs ParseListener::getLocations( antlr4::ParserRuleContext *ctx )
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

        mlir::FileLineColLoc startLoc =
            mlir::FileLineColLoc::get( builder.getStringAttr( filename ), startLine, startCol + 1 );
        mlir::FileLineColLoc endLoc =
            mlir::FileLineColLoc::get( builder.getStringAttr( filename ), endLine, endCol + 1 );

        return { startLoc, endLoc };
    }

    inline mlir::Location ParseListener::getStartLocation( antlr4::ParserRuleContext *ctx )
    {
        LocPairs locs = getLocations( ctx );

        return locs.first;
    }

    inline mlir::Location ParseListener::getStopLocation( antlr4::ParserRuleContext *ctx )
    {
        LocPairs locs = getLocations( ctx );

        return locs.second;
    }

#if 0
    inline mlir::Location ParseListener::getTerminalLocation( antlr4::tree::TerminalNode *node )
    {
        assert( node );
        antlr4::Token *token = node->getSymbol();

        assert( token );
        size_t line = token->getLine();
        size_t column = token->getCharPositionInLine() + 1;

        return mlir::FileLineColLoc::get( builder.getStringAttr( filename ), line, column );
    }
#endif

    DialectCtx::DialectCtx()
    {
        context.getOrLoadDialect<silly::SillyDialect>();
        context.getOrLoadDialect<mlir::func::FuncDialect>();
        context.getOrLoadDialect<mlir::arith::ArithDialect>();
        context.getOrLoadDialect<mlir::memref::MemRefDialect>();
        context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
        context.getOrLoadDialect<mlir::scf::SCFDialect>();
    }

    bool ParseListener::isVariableDeclared( mlir::Location loc, const std::string &varName )
    {
        // Get the single scope
        PerFunctionState &f = funcState( currentFuncName );

        silly::ScopeOp scopeOp = getEnclosingScopeOp( loc, f.getFuncOp() );

        mlir::Operation *symbolOp = mlir::SymbolTable::lookupSymbolIn( scopeOp, varName );
        return ( symbolOp != nullptr );
    }

    inline mlir::Value ParseListener::parseExpression( SillyParser::ExpressionContext *ctx, mlir::Type ty )
    {
        assert( ctx );
        mlir::Value value = parseLowest( ctx, ty );
        if ( ty )
        {
            mlir::Location loc = getStartLocation( ctx );

            value = castOpIfRequired( loc, value, ty );
        }

        return value;
    }

    void ParseListener::registerDeclaration( mlir::Location loc, const std::string &varName, mlir::Type ty,
                                             SillyParser::ArrayBoundsExpressionContext *arrayBounds,
                                             SillyParser::ExpressionContext *assignmentExpression,
                                             const std::vector<SillyParser::ExpressionContext *> *pExpressions )
    {
        size_t arraySize{};
        size_t numElements{ 1 };
        if ( arrayBounds )
        {
            tNode *index = arrayBounds->INTEGER_PATTERN();
            assert( index );
            arraySize = std::stoi( index->getText() );
            numElements = arraySize;
        }

        mlir::OpBuilder::InsertPoint savedIP = builder.saveInsertionPoint();

        PerFunctionState &f = funcState( currentFuncName );

        mlir::func::FuncOp funcOp = f.getFuncOp();

        silly::ScopeOp scopeOp = getEnclosingScopeOp( loc, funcOp );

        mlir::Operation *symbolOp = mlir::SymbolTable::lookupSymbolIn( scopeOp, varName );
        if ( symbolOp )
        {
            throw UserError( loc, std::format( "Variable {} already declared", varName ) );
        }

        if ( f.lastDeclareOp )
        {
            builder.setInsertionPointAfter( f.lastDeclareOp );
        }
        else
        {
            // Scope has one block
            mlir::Block *scopeBlock = &scopeOp.getBody().front();

            // Insert declarations at the beginning of the scope block
            // (all DeclareOps should appear before any scf.if/scf.for)
            builder.setInsertionPointToStart( scopeBlock );
        }

        std::vector<mlir::Value> initializers;

        if ( pExpressions )
        {
            mlir::Value fill{};

            for ( SillyParser::ExpressionContext *e : *pExpressions )
            {
                initializers.push_back( parseExpression( e, ty ) );
            }

            ssize_t remaining = numElements - initializers.size();

            if ( remaining )
            {
                if ( ty == tyI1 )
                {
                    fill = parseBoolean( loc, "FALSE" );
                }
                else if ( mlir::IntegerType ity = mlir::dyn_cast<mlir::IntegerType>( ty ) )
                {
                    int width = ity.getWidth();
                    fill = parseInteger( loc, width, "0" );
                }
                else if ( mlir::FloatType fty = mlir::dyn_cast<mlir::FloatType>( ty ) )
                {
                    fill = parseFloat( loc, fty, "0" );
                }
                else
                {
                    throw ExceptionWithContext(
                        __FILE__, __LINE__, __func__,
                        std::format( "{}internal error: unknown scalar type.\n", formatLocation( loc ) ) );
                }
            }

            if ( initializers.size() > numElements )
            {
                throw UserError(
                    loc,
                    std::format( "For variable '{}', more initializers ({}) specified than number of elements ({}).\n",
                                 varName, initializers.size(), numElements ) );
            }

            for ( ssize_t i = 0; i < remaining; i++ )
            {
                assert( fill );
                initializers.push_back( fill );
            }
        }

        mlir::StringAttr strAttr = builder.getStringAttr( varName );
        silly::DeclareOp dcl;
        if ( arraySize )
        {
            dcl = builder.create<silly::DeclareOp>( loc, mlir::TypeAttr::get( ty ),
                                                    builder.getI64IntegerAttr( arraySize ),
                                                    /*parameter=*/nullptr, nullptr, initializers );
        }
        else
        {
            dcl = builder.create<silly::DeclareOp>( loc, mlir::TypeAttr::get( ty ), nullptr, /*parameter=*/nullptr,
                                                    nullptr, initializers );
        }
        dcl->setAttr( "sym_name", strAttr );

        f.lastDeclareOp = dcl.getOperation();

        builder.restoreInsertionPoint( savedIP );

        if ( assignmentExpression )
        {
            processAssignment( loc, assignmentExpression, varName, {} );
        }
    }

    inline mlir::Value ParseListener::parseLowest( antlr4::ParserRuleContext *ctx, mlir::Type ty )
    {
        SillyParser::ExprLowestContext *expr = dynamic_cast<SillyParser::ExprLowestContext *>( ctx );
        if ( !expr )
        {
            mlir::Location loc = getStartLocation( ctx );

            throw ExceptionWithContext(
                __FILE__, __LINE__, __func__,
                std::format( "{}internal error: unexpected ExpressionContext alternative: {}.\n", formatLocation( loc ),
                             ctx->getText() ) );
        }

        SillyParser::BinaryExpressionLowestContext *lowest = expr->binaryExpressionLowest();

        return parseOr( lowest, ty );
    }

    void ParseListener::syntaxError( antlr4::Recognizer *recognizer, antlr4::Token *offendingSymbol, size_t line,
                                     size_t charPositionInLine, const std::string &msg, std::exception_ptr e )
    {
        hasErrors = true;
        std::string tokenText = offendingSymbol ? offendingSymbol->getText() : "<none>";
        throw ExceptionWithContext( __FILE__, __LINE__, __func__,
                                    std::format( "Syntax error in {}:{}:{}: {} (token: {} )", filename, line,
                                                 charPositionInLine, msg, tokenText ) );
    }

    silly::ScopeOp getEnclosingScopeOp( mlir::Location loc, mlir::func::FuncOp funcOp )
    {
        // Single ScopeOp per function – iterate once to find it
        for ( mlir::Operation &op : funcOp.getBody().front() )
        {
            if ( silly::ScopeOp scopeOp = dyn_cast<silly::ScopeOp>( &op ) )
            {
                return scopeOp;
            }
        }

        throw ExceptionWithContext(
            __FILE__, __LINE__, __func__,
            std::format( "{}internal error: Unable to find Enclosing ScopeOp\n", formatLocation( loc ) ) );

        return nullptr;
    }

    silly::DeclareOp ParseListener::lookupDeclareForVar( mlir::Location loc, const std::string &varName )
    {
        PerFunctionState &f = funcState( currentFuncName );
        mlir::func::FuncOp funcOp = f.getFuncOp();

        silly::ScopeOp scopeOp = getEnclosingScopeOp( loc, funcOp );

        // LLVM_DEBUG( {
        //     llvm::errs() << std::format( "Lookup symbol {} in parent function:\n", varName );
        //     scopeOp->dump();
        // } );

        mlir::Operation *symbolOp = mlir::SymbolTable::lookupSymbolIn( scopeOp, varName );
        if ( !symbolOp )
        {
            throw UserError( loc, std::format( "Undeclared variable {} (symbol lookup failed.)", varName ) );
        }

        silly::DeclareOp declareOp = mlir::dyn_cast<silly::DeclareOp>( symbolOp );
        if ( !declareOp )
        {
            throw UserError( loc, std::format( "Undeclared variable {} (DeclareOp not found)", varName ) );
        }

        return declareOp;
    }

    mlir::Type ParseListener::parseScalarType( const std::string &ty )
    {
        if ( ty == "BOOL" )
        {
            return tyI1;
        }
        if ( ty == "INT8" )
        {
            return tyI8;
        }
        if ( ty == "INT16" )
        {
            return tyI16;
        }
        if ( ty == "INT32" )
        {
            return tyI32;
        }
        if ( ty == "INT64" )
        {
            return tyI64;
        }
        if ( ty == "FLOAT32" )
        {
            return tyF32;
        }
        if ( ty == "FLOAT64" )
        {
            return tyF64;
        }
        return nullptr;
    }

    ParseListener::ParseListener( const std::string &filenameIn )
        : filename( filenameIn ),
          dialect(),
          builder( &dialect.context ),
          mod( mlir::ModuleOp::create( getStartLocation( nullptr ) ) )
    {
        builder.setInsertionPointToStart( mod.getBody() );

        tyI1 = builder.getI1Type();
        tyI8 = builder.getI8Type();
        tyI16 = builder.getI16Type();
        tyI32 = builder.getI32Type();
        tyI64 = builder.getI64Type();

        tyF32 = builder.getF32Type();
        tyF64 = builder.getF64Type();

        mlir::MLIRContext *ctx = builder.getContext();
        tyVoid = mlir::LLVM::LLVMVoidType::get( ctx );
        tyPtr = mlir::LLVM::LLVMPointerType::get( ctx );
    }

    void ParseListener::createScope( mlir::Location startLoc, mlir::Location endLoc, mlir::func::FuncOp funcOp,
                                     const std::string &funcName, const std::vector<std::string> &paramNames )
    {
        LLVM_DEBUG( {
            llvm::errs() << std::format( "createScope: {}: startLoc: {}, endLoc: {}\n", funcName,
                                         formatLocation( startLoc ), formatLocation( endLoc ) );
        } );
        mlir::Block &block = *funcOp.addEntryBlock();
        builder.setInsertionPointToStart( &block );

        // initially with empty operands and results
        silly::ScopeOp scopeOp = builder.create<silly::ScopeOp>( startLoc, mlir::TypeRange{}, mlir::ValueRange{} );
        builder.create<silly::YieldOp>( endLoc );

        mlir::Block &scopeBlock = scopeOp.getBody().emplaceBlock();

        builder.setInsertionPointToStart( &scopeBlock );

        PerFunctionState &f = funcState( funcName );

        for ( size_t i = 0; i < funcOp.getNumArguments() && i < paramNames.size(); ++i )
        {
            mlir::Type argType = funcOp.getArgument( i ).getType();
            LLVM_DEBUG( {
                llvm::errs() << std::format( "function {}: parameter{}:\n", funcName, i );
                funcOp.getArgument( i ).dump();
            } );
            mlir::StringAttr strAttr = builder.getStringAttr( paramNames[i] );
            silly::DeclareOp dcl = builder.create<silly::DeclareOp>(
                startLoc, mlir::TypeAttr::get( argType ), /*size=*/nullptr, builder.getUnitAttr(),
                builder.getI64IntegerAttr( i ), mlir::ValueRange{} );
            dcl->setAttr( "sym_name", strAttr );

            f.lastDeclareOp = dcl.getOperation();
        }

        currentFuncName = funcName;
        f.setFuncOp( funcOp );
    }

    void ParseListener::enterStartRule( SillyParser::StartRuleContext *ctx )
    try
    {
        assert( ctx );
        currentFuncName = ENTRY_SYMBOL_NAME;

        LocPairs locs = getLocations( ctx );

        mlir::FunctionType funcType = builder.getFunctionType( {}, tyI32 );
        mlir::func::FuncOp funcOp = builder.create<mlir::func::FuncOp>( locs.first, ENTRY_SYMBOL_NAME, funcType );

        std::vector<std::string> paramNames;
        createScope( locs.first, locs.second, funcOp, ENTRY_SYMBOL_NAME, paramNames );
    }
    CATCH_USER_ERROR

    void ParseListener::processReturnLike( mlir::Location loc, SillyParser::ExpressionContext *expression )
    {
        mlir::Type returnType{};
        mlir::Value value{};

        if ( currentFuncName == ENTRY_SYMBOL_NAME )
        {
            returnType = tyI32;
        }
        else
        {
            PerFunctionState &f = funcState( currentFuncName );
            mlir::func::FuncOp funcOp = f.getFuncOp();
            llvm::ArrayRef<mlir::Type> returnTypeArray = funcOp.getFunctionType().getResults();

            if ( !returnTypeArray.empty() )
            {
                returnType = returnTypeArray[0];
            }
        }

        if ( expression )
        {
            if ( !returnType )
            {
                throw UserError( loc, std::format( "return expression found '{}', but no return type for function {}",
                                                   expression->getText(), currentFuncName ) );
            }

            value = parseExpression( expression, returnType );
        }
        else if ( currentFuncName == ENTRY_SYMBOL_NAME )
        {
            value = builder.create<mlir::arith::ConstantIntOp>( loc, 0, 32 );
        }

        // Create new ReturnOp with user specified value:
        if ( value )
        {
            builder.create<silly::ReturnOp>( loc, mlir::ValueRange{ value } );
        }
        else
        {
            builder.create<silly::ReturnOp>( loc, mlir::ValueRange{} );
        }
    }

    void ParseListener::exitStartRule( SillyParser::StartRuleContext *ctx )
    try
    {
        assert( ctx );
        mlir::Location loc = getStartLocation( ctx );

        assert( currentFuncName == ENTRY_SYMBOL_NAME );

        if ( !ctx->exitStatement() )
        {
            processReturnLike( loc, nullptr );
        }

        LLVM_DEBUG( {
            llvm::errs() << "exitStartRule done: module dump:\n";
            mod->dump();
        } );
    }
    CATCH_USER_ERROR

    void ParseListener::enterFunctionStatement( SillyParser::FunctionStatementContext *ctx )
    try
    {
        assert( ctx );
        LocPairs locs = getLocations( ctx );

        LLVM_DEBUG( {
            llvm::errs() << std::format( "enterFunctionStatement: startLoc: {}, endLoc: {}:\n",
                                         formatLocation( locs.first ), formatLocation( locs.second ) );
        } );

        mainIP = builder.saveInsertionPoint();

        builder.setInsertionPointToStart( mod.getBody() );

        assert( ctx->IDENTIFIER() );
        std::string funcName = ctx->IDENTIFIER()->getText();

        std::vector<mlir::Type> returns;
        if ( SillyParser::ScalarTypeContext *rt = ctx->scalarType() )
        {
            mlir::Type returnType = parseScalarType( rt->getText() );
            returns.push_back( returnType );
        }

        std::vector<mlir::Type> paramTypes;
        std::vector<std::string> paramNames;
        for ( SillyParser::VariableTypeAndNameContext *paramCtx : ctx->variableTypeAndName() )
        {
            assert( paramCtx->scalarType() );
            mlir::Type paramType = parseScalarType( paramCtx->scalarType()->getText() );
            assert( paramCtx->IDENTIFIER() );
            std::string paramName = paramCtx->IDENTIFIER()->getText();
            paramTypes.push_back( paramType );
            paramNames.push_back( paramName );
        }

        std::vector<mlir::NamedAttribute> attrs;
        attrs.push_back(
            mlir::NamedAttribute( builder.getStringAttr( "sym_visibility" ), builder.getStringAttr( "private" ) ) );

        mlir::FunctionType funcType = builder.getFunctionType( paramTypes, returns );
        mlir::func::FuncOp funcOp = builder.create<mlir::func::FuncOp>( locs.first, funcName, funcType, attrs );
        createScope( locs.first, locs.second, funcOp, funcName, paramNames );
    }
    CATCH_USER_ERROR

    void ParseListener::exitFunctionStatement( SillyParser::FunctionStatementContext *ctx )
    try
    {
        assert( ctx );
        builder.restoreInsertionPoint( mainIP );

        currentFuncName = ENTRY_SYMBOL_NAME;
    }
    CATCH_USER_ERROR

    mlir::Value ParseListener::handleCall( SillyParser::CallExpressionContext *ctx )
    {
        assert( ctx );
        mlir::Location loc = getStartLocation( ctx );
        tNode *id = ctx->IDENTIFIER();
        assert( id );
        std::string funcName = id->getText();
        PerFunctionState &f = funcState( funcName );
        mlir::func::FuncOp funcOp = f.getFuncOp();
        mlir::FunctionType funcType = funcOp.getFunctionType();
        std::vector<mlir::Value> parameters;

        if ( SillyParser::ParameterListContext *params = ctx->parameterList() )
        {
            int i = 0;

            assert( params );
            size_t psz = params->parameterExpression().size();
            size_t fsz = funcType.getInputs().size();
            assert( psz == fsz );

            for ( SillyParser::ParameterExpressionContext *e : params->parameterExpression() )
            {
                SillyParser::ExpressionContext *p = e->expression();
                assert( p );
                std::string paramText = p->getText();
                std::cout << std::format( "CALL function {}: param: {}\n", funcName, paramText );

                mlir::Type ty = funcType.getInputs()[i];
                mlir::Value value = parseExpression( p, ty );

                parameters.push_back( value );
                i++;
            }
        }

        mlir::TypeRange resultTypes = funcType.getResults();

        silly::CallOp callOp = builder.create<silly::CallOp>( loc, resultTypes, funcName, parameters );

        // Return the first result (or null for void calls)
        return resultTypes.empty() ? mlir::Value{} : callOp.getResults()[0];
    }

    void ParseListener::enterCallStatement( SillyParser::CallStatementContext *ctx )
    try
    {
        assert( ctx );
        handleCall( ctx->callExpression() );
    }
    CATCH_USER_ERROR

    void ParseListener::enterDeclareHelper(
        mlir::Location loc, tNode *identifier,
        SillyParser::DeclareAssignmentExpressionContext *declareAssignmentExpression,
        const std::vector<SillyParser::ExpressionContext *> &expressions, tNode *hasInitList,
        SillyParser::ArrayBoundsExpressionContext *arrayBoundsExpression, mlir::Type ty )
    try
    {
        assert( identifier );
        std::string varName = identifier->getText();

        const std::vector<SillyParser::ExpressionContext *> *pExpressions{};
        SillyParser::ExpressionContext *assignmentExpression{};

        if ( hasInitList || expressions.size() )
        {
            if ( declareAssignmentExpression )
            {
                throw UserError(
                    loc,
                    std::format(
                        "Declaration cannot have both assignment expression and initialization-list expression." ) );
            }
            pExpressions = &expressions;
        }
        else if ( declareAssignmentExpression )
        {
            assignmentExpression = declareAssignmentExpression->expression();
        }

        registerDeclaration( loc, varName, ty, arrayBoundsExpression, assignmentExpression, pExpressions );

        // LLVM_DEBUG( { llvm::errs() << "enterDeclareHelper done: module dump:\n"; mod->dump(); } );
    }
    CATCH_USER_ERROR

    void ParseListener::enterDeclareStatement( SillyParser::DeclareStatementContext *ctx )
    {
        assert( ctx );
        mlir::Location loc = getStartLocation( ctx );

        enterDeclareHelper( loc, ctx->IDENTIFIER(), ctx->declareAssignmentExpression(), ctx->expression(),
                            ctx->LEFT_CURLY_BRACKET_TOKEN(), ctx->arrayBoundsExpression(), tyF64 );
    }

    void ParseListener::enterBoolDeclareStatement( SillyParser::BoolDeclareStatementContext *ctx )
    {
        assert( ctx );
        mlir::Location loc = getStartLocation( ctx );
        enterDeclareHelper( loc, ctx->IDENTIFIER(), ctx->declareAssignmentExpression(), ctx->expression(),
                            ctx->LEFT_CURLY_BRACKET_TOKEN(), ctx->arrayBoundsExpression(), tyI1 );
    }

    mlir::Type ParseListener::integerDeclarationType( mlir::Location loc, SillyParser::IntTypeContext * ctx )
    {
        mlir::Type ty;

        if ( ctx->INT8_TOKEN() )
        {
            ty = tyI8;
        }
        else if ( ctx->INT16_TOKEN() )
        {
            ty = tyI16;
        }
        else if ( ctx->INT32_TOKEN() )
        {
            ty = tyI32;
        }
        else if ( ctx->INT64_TOKEN() )
        {
            ty = tyI64;
        }
        else
        {
            throw ExceptionWithContext( __FILE__, __LINE__, __func__,
                                        std::format( "{}internal error: Unsupported signed integer declaration size.\n",
                                                     formatLocation( loc ) ) );
        }

        return ty;
    }

    void ParseListener::enterIntDeclareStatement( SillyParser::IntDeclareStatementContext *ctx )
    {
        assert( ctx );
        mlir::Location loc = getStartLocation( ctx );

        mlir::Type ty = integerDeclarationType( loc, ctx->intType() );

        enterDeclareHelper( loc, ctx->IDENTIFIER(), ctx->declareAssignmentExpression(), ctx->expression(),
                            ctx->LEFT_CURLY_BRACKET_TOKEN(), ctx->arrayBoundsExpression(), ty );
    }

    void ParseListener::enterFloatDeclareStatement( SillyParser::FloatDeclareStatementContext *ctx )
    {
        assert( ctx );
        mlir::Location loc = getStartLocation( ctx );
        mlir::Type ty;

        if ( ctx->FLOAT32_TOKEN() )
        {
            ty = tyF32;
        }
        else if ( ctx->FLOAT64_TOKEN() )
        {
            ty = tyF64;
        }
        else
        {
            mlir::Location loc = getStartLocation( ctx );
            throw ExceptionWithContext( __FILE__, __LINE__, __func__,
                                        std::format( "{}internal error: Unsupported floating point declaration size.\n",
                                                     formatLocation( loc ) ) );
        }

        enterDeclareHelper( loc, ctx->IDENTIFIER(), ctx->declareAssignmentExpression(), ctx->expression(),
                            ctx->LEFT_CURLY_BRACKET_TOKEN(), ctx->arrayBoundsExpression(), ty );
    }

    void ParseListener::enterStringDeclareStatement( SillyParser::StringDeclareStatementContext *ctx )
    try
    {
        assert( ctx );
        mlir::Location loc = getStartLocation( ctx );
        assert( ctx->IDENTIFIER() );
        std::string varName = ctx->IDENTIFIER()->getText();
        SillyParser::ArrayBoundsExpressionContext *arrayBounds = ctx->arrayBoundsExpression();
        assert( arrayBounds );

        registerDeclaration( loc, varName, tyI8, arrayBounds, nullptr, nullptr );

        if ( tNode *theString = ctx->STRING_PATTERN() )
        {
            std::string s = stripQuotes( loc, theString->getText() );

            mlir::SymbolRefAttr symRef = mlir::SymbolRefAttr::get( &dialect.context, varName );
            mlir::StringAttr strAttr = builder.getStringAttr( s );

            silly::StringLiteralOp stringLiteral = builder.create<silly::StringLiteralOp>( loc, tyPtr, strAttr );

            mlir::NamedAttribute varNameAttr( builder.getStringAttr( "var_name" ), symRef );

            builder.create<silly::AssignOp>( loc, mlir::TypeRange{}, mlir::ValueRange{ stringLiteral },
                                             llvm::ArrayRef<mlir::NamedAttribute>{ varNameAttr } );
        }
    }
    CATCH_USER_ERROR

    void ParseListener::createIf( mlir::Location loc, SillyParser::ExpressionContext *predicate, bool saveIP )
    {
        mlir::Value conditionPredicate = parseExpression( predicate, {} );

        mlir::scf::IfOp ifOp = builder.create<mlir::scf::IfOp>( loc, conditionPredicate,
                                                                /*withElseRegion=*/true );

        if ( saveIP )
        {
            insertionPointStack.push_back( ifOp.getOperation() );
        }

        mlir::Block &thenBlock = ifOp.getThenRegion().front();
        builder.setInsertionPointToStart( &thenBlock );
    }

    void ParseListener::enterIfStatement( SillyParser::IfStatementContext *ctx )
    try
    {
        assert( ctx );
        mlir::Location loc = getStartLocation( ctx );

        createIf( loc, ctx->expression(), true );
    }
    CATCH_USER_ERROR

    void ParseListener::selectElseBlock( mlir::Location loc, const std::string &errorText )
    {
        mlir::Block *currentBlock = builder.getInsertionBlock();
        assert( currentBlock );

        // Get the parent region of the current block (the then region).
        mlir::Region *parentRegion = currentBlock->getParent();

        // Verify it's inside an scf.if by checking the parent op.
        mlir::Operation *parentOp = parentRegion->getParentOp();
        mlir::scf::IfOp ifOp = dyn_cast<mlir::scf::IfOp>( parentOp );

        if ( !ifOp )
        {
            throw ExceptionWithContext(
                __FILE__, __LINE__, __func__,
                std::format( "{}internal error: Current insertion point must be inside an scf.if then region\n",
                             formatLocation( loc ), errorText ) );
        }

        // Set the insertion point to the start of the else region's (first) block.
        mlir::Region &elseRegion = ifOp.getElseRegion();
        mlir::Block &elseBlock = elseRegion.front();
        builder.setInsertionPointToStart( &elseBlock );
    }

    void ParseListener::enterElseStatement( SillyParser::ElseStatementContext *ctx )
    try
    {
        assert( ctx );
        mlir::Location loc = getStartLocation( ctx );

        selectElseBlock( loc, ctx->getText() );
    }
    CATCH_USER_ERROR

    void ParseListener::enterElifStatement( SillyParser::ElifStatementContext *ctx )
    try
    {
        assert( ctx );
        mlir::Location loc = getStartLocation( ctx );

        selectElseBlock( loc, ctx->getText() );

        createIf( loc, ctx->expression(), false );
    }
    CATCH_USER_ERROR

    void ParseListener::exitIfElifElseStatement( SillyParser::IfElifElseStatementContext *ctx )
    try
    {
        // Restore EXACTLY where we were before creating the scf.if
        // This places new ops right AFTER the scf.if
        builder.setInsertionPointAfter( insertionPointStack.back() );
        insertionPointStack.pop_back();
    }
    CATCH_USER_ERROR

    void ParseListener::enterForStatement( SillyParser::ForStatementContext *ctx )
    try
    {
        assert( ctx );
        mlir::Location loc = getStartLocation( ctx );

        LLVM_DEBUG( { llvm::errs() << std::format( "For: {}\n", ctx->getText() ); } );

        assert( ctx->IDENTIFIER() );
        std::string varName = ctx->IDENTIFIER()->getText();

        mlir::SymbolRefAttr symRef = mlir::SymbolRefAttr::get( &dialect.context, varName );
        mlir::NamedAttribute varNameAttr( builder.getStringAttr( "var_name" ), symRef );

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

        if ( isVariableDeclared( loc, varName ) )
        {
            throw UserError( loc, std::format( "Induction variable {} clashes with declared variable in: {}\n", varName, ctx->getText() ) );
        }

        mlir::Value p = searchForInduction( varName );
        if ( p )
        {
            throw UserError( loc, std::format( "Induction variable {} used by enclosing FOR: {}\n", varName, ctx->getText() ) );
        }

        mlir::Type elemType = integerDeclarationType( loc, ctx->intType() );

        std::string s;
        if ( pStart )
        {
            start = parseExpression( pStart, elemType );
        }
        else
        {
            throw ExceptionWithContext( __FILE__, __LINE__, __func__,
                                        std::format( "{}internal error: FOR loop: expected start index: {}\n",
                                                     formatLocation( loc ), ctx->getText() ) );
        }

        if ( pEnd )
        {
            end = parseExpression( pEnd, elemType );
        }
        else
        {
            throw ExceptionWithContext( __FILE__, __LINE__, __func__,
                                        std::format( "{}internal error: FOR loop: expected end index: {}\n",
                                                     formatLocation( loc ), ctx->getText() ) );
        }

        if ( pStep )
        {
            step = parseExpression( pStep, elemType );
        }
        else
        {
            mlir::IntegerType ity = mlir::dyn_cast<mlir::IntegerType>( elemType );
            unsigned width = ity.getWidth();

            //'scf.for' op failed to verify that all of {lowerBound, upperBound, step} have same type
            step = builder.create<mlir::arith::ConstantIntOp>( loc, 1, width );
            step = castOpIfRequired( loc, step, elemType );
        }

        mlir::scf::ForOp forOp = builder.create<mlir::scf::ForOp>( loc, start, end, step );
        insertionPointStack.push_back( forOp.getOperation() );

        mlir::Block &loopBody = forOp.getRegion().front();
        builder.setInsertionPointToStart( &loopBody );

        mlir::Value inductionVar = loopBody.getArgument( 0 );
        pushInductionVariable( varName, inductionVar );
    }
    CATCH_USER_ERROR

    void ParseListener::exitForStatement( SillyParser::ForStatementContext *ctx )
    try
    {
        assert( ctx );
        builder.setInsertionPointAfter( insertionPointStack.back() );
        insertionPointStack.pop_back();
        popInductionVariable();
    }
    CATCH_USER_ERROR

    void ParseListener::handlePrint( mlir::Location loc, const std::vector<SillyParser::ExpressionContext *> &args,
                                     const std::string &errorContextString, PrintFlags pf )
    {
        std::vector<mlir::Value> vargs;
        for ( SillyParser::ExpressionContext *parg : args )
        {
            mlir::Value v = parseExpression( parg, {} );

            vargs.push_back( v );
        }

        mlir::arith::ConstantIntOp constFlagOp = builder.create<mlir::arith::ConstantIntOp>( loc, pf, 32 );
        builder.create<silly::PrintOp>( loc, constFlagOp, vargs );
    }

    void ParseListener::enterPrintStatement( SillyParser::PrintStatementContext *ctx )
    try
    {
        assert( ctx );
        mlir::Location loc = getStartLocation( ctx );

        int flags = PRINT_FLAGS_NONE;
        if ( ctx->CONTINUE_TOKEN() )
        {
            flags = PRINT_FLAGS_CONTINUE;
        }
        handlePrint( loc, ctx->expression(), ctx->getText(), (PrintFlags)flags );
    }
    CATCH_USER_ERROR

    void ParseListener::enterErrorStatement( SillyParser::ErrorStatementContext *ctx )
    try
    {
        assert( ctx );
        mlir::Location loc = getStartLocation( ctx );
        int flags = PRINT_FLAGS_ERROR;
        if ( ctx->CONTINUE_TOKEN() )
        {
            flags |= PRINT_FLAGS_CONTINUE;
        }
        handlePrint( loc, ctx->expression(), ctx->getText(), (PrintFlags)flags );
    }
    CATCH_USER_ERROR

    void ParseListener::enterAbortStatement( SillyParser::AbortStatementContext *ctx )
    try
    {
        assert( ctx );
        mlir::Location loc = getStartLocation( ctx );

        builder.create<silly::AbortOp>( loc );
    }
    CATCH_USER_ERROR

    void ParseListener::enterGetStatement( SillyParser::GetStatementContext *ctx )
    try
    {
        assert( ctx );
        mlir::Location loc = getStartLocation( ctx );

        SillyParser::ScalarOrArrayElementContext *scalarOrArrayElement = ctx->scalarOrArrayElement();
        if ( scalarOrArrayElement )
        {
            tNode *varNameObject = scalarOrArrayElement->IDENTIFIER();
            assert( varNameObject );
            std::string varName = varNameObject->getText();

            silly::DeclareOp declareOp = lookupDeclareForVar( loc, varName );

            mlir::Type elemType = declareOp.getTypeAttr().getValue();
            mlir::Value optIndexValue{};
            if ( SillyParser::IndexExpressionContext *indexExpr = scalarOrArrayElement->indexExpression() )
            {
                mlir::Value indexValue = parseExpression( indexExpr->expression(), {} );

                mlir::Location iloc = getStartLocation( indexExpr->expression() );
                optIndexValue = indexTypeCast( iloc, indexValue );
            }
            else if ( declareOp.getSizeAttr() )
            {
                throw UserError( loc, std::format( "Attempted GET to string literal?: {}", ctx->getText() ) );
            }
            else
            {
                // Scalar: load the value
            }

            mlir::SymbolRefAttr symRef = mlir::SymbolRefAttr::get( &dialect.context, varName );

            silly::GetOp resultValue = builder.create<silly::GetOp>( loc, elemType );
            builder.create<silly::AssignOp>( loc, symRef, optIndexValue, resultValue );
        }
        else
        {
            throw ExceptionWithContext(
                __FILE__, __LINE__, __func__,
                std::format( "{}internal error: unexpected get context {}\n", formatLocation( loc ), ctx->getText() ) );
        }
    }
    CATCH_USER_ERROR

    mlir::Type ParseListener::biggestTypeOf( mlir::Type ty1, mlir::Type ty2 )
    {
        if ( ty1 == ty2 )
        {
            return ty1;
        }
        else if ( ty1.isF64() )
        {
            return ty1;
        }
        else if ( ty2.isF64() )
        {
            return ty2;
        }
        else if ( ty1.isF32() )
        {
            return ty1;
        }
        else if ( ty2.isF32() )
        {
            return ty2;
        }
        else
        {
            mlir::IntegerType ity1 = mlir::cast<mlir::IntegerType>( ty1 );
            mlir::IntegerType ity2 = mlir::cast<mlir::IntegerType>( ty2 );

            unsigned w1 = ity1.getWidth();
            unsigned w2 = ity2.getWidth();

            if ( w1 > w2 )
            {
                return ty1;
            }
            else
            {
                return ty2;
            }
        }
    }

    mlir::Value ParseListener::castOpIfRequired( mlir::Location loc, mlir::Value value, mlir::Type desiredType )
    {
        if ( value.getType() != desiredType )
        {
            mlir::Type valType = value.getType();

            if ( valType.isF64() )
            {
                if ( mlir::isa<mlir::IntegerType>( desiredType ) )
                {
                    value = builder.create<mlir::arith::FPToSIOp>( loc, desiredType, value );
                }
                else if ( desiredType.isF32() )
                {
                    value = builder.create<mlir::LLVM::FPExtOp>( loc, desiredType, value );
                }
            }
            else if ( valType.isF32() )
            {
                if ( mlir::isa<mlir::IntegerType>( desiredType ) )
                {
                    value = builder.create<mlir::arith::FPToSIOp>( loc, desiredType, value );
                }
                else if ( desiredType.isF64() )
                {
                    value = builder.create<mlir::LLVM::FPExtOp>( loc, desiredType, value );
                }
            }
            else if ( mlir::IntegerType viType = mlir::cast<mlir::IntegerType>( valType ) )
            {
                unsigned vwidth = viType.getWidth();
                if ( mlir::isa<mlir::FloatType>( desiredType ) )
                {
                    if ( vwidth == 1 )
                    {
                        value = builder.create<mlir::arith::UIToFPOp>( loc, desiredType, value );
                    }
                    else
                    {
                        value = builder.create<mlir::arith::SIToFPOp>( loc, desiredType, value );
                    }
                }
                else if ( mlir::IntegerType miType = mlir::cast<mlir::IntegerType>( desiredType ) )
                {
                    unsigned mwidth = miType.getWidth();
                    if ( ( vwidth == 1 ) && ( mwidth != 1 ) )
                    {
                        // widen bool to integer using unsigned extension:
                        value = builder.create<mlir::arith::ExtUIOp>( loc, desiredType, value );
                    }
                    else if ( vwidth > mwidth )
                    {
                        value = builder.create<mlir::arith::TruncIOp>( loc, desiredType, value );
                    }
                    else if ( vwidth < mwidth )
                    {
                        value = builder.create<mlir::arith::ExtSIOp>( loc, desiredType, value );
                    }
                }
            }
        }

        return value;
    }

    void ParseListener::enterReturnStatement( SillyParser::ReturnStatementContext *ctx )
    try
    {
        assert( ctx );
        mlir::Location loc = getStartLocation( ctx );

        processReturnLike( loc, ctx->expression() );
    }
    CATCH_USER_ERROR

    void ParseListener::enterExitStatement( SillyParser::ExitStatementContext *ctx )
    try
    {
        assert( ctx );
        mlir::Location loc = getStartLocation( ctx );

        processReturnLike( loc, ctx->expression() );
    }
    CATCH_USER_ERROR

    void ParseListener::processAssignment( mlir::Location loc, SillyParser::ExpressionContext *exprContext,
                                           const std::string &currentVarName, mlir::Value currentIndexExpr )
    {
        mlir::Value resultValue = parseExpression( exprContext, {} );

        mlir::SymbolRefAttr symRef = mlir::SymbolRefAttr::get( &dialect.context, currentVarName );

        assert( resultValue );

        mlir::BlockArgument ba = mlir::dyn_cast<mlir::BlockArgument>( resultValue );
        mlir::Operation * op = resultValue.getDefiningOp();

        // Don't check if it's a StringLiteralOp if it's an induction variable, since op will be nullptr
        if ( !ba && isa<silly::StringLiteralOp>( op ) )
        {
            mlir::NamedAttribute varNameAttr( builder.getStringAttr( "var_name" ), symRef );

            builder.create<silly::AssignOp>( loc, mlir::TypeRange{}, mlir::ValueRange{ resultValue },
                                             llvm::ArrayRef<mlir::NamedAttribute>{ varNameAttr } );
        }
        else
        {
            if ( currentIndexExpr )
            {
                mlir::Value i = indexTypeCast( loc, currentIndexExpr );

                silly::AssignOp assign = builder.create<silly::AssignOp>( loc, symRef, i, resultValue );

                LLVM_DEBUG( {
                    mlir::OpPrintingFlags flags;
                    flags.enableDebugInfo( true );

                    assign->print( llvm::outs(), flags );
                    llvm::outs() << "\n";
                } );
            }
            else
            {
                mlir::NamedAttribute varNameAttr( builder.getStringAttr( "var_name" ), symRef );

                builder.create<silly::AssignOp>( loc, mlir::TypeRange{}, mlir::ValueRange{ resultValue },
                                                 llvm::ArrayRef<mlir::NamedAttribute>{ varNameAttr } );
            }
        }
    }

    void ParseListener::enterAssignmentStatement( SillyParser::AssignmentStatementContext *ctx )
    try
    {
        assert( ctx );
        mlir::Location loc = getStartLocation( ctx );
        SillyParser::ScalarOrArrayElementContext *lhs = ctx->scalarOrArrayElement();
        assert( lhs );
        assert( lhs->IDENTIFIER() );
        std::string currentVarName = lhs->IDENTIFIER()->getText();

        SillyParser::IndexExpressionContext *indexExpr = lhs->indexExpression();
        mlir::Value currentIndexExpr = mlir::Value{};

        if ( !isVariableDeclared( loc, currentVarName ) )
        {
            throw UserError( loc, std::format( "Attempt to assign to undeclared variable: {}\n", ctx->getText() ) );
        }

        if ( indexExpr )
        {
            currentIndexExpr = parseExpression( indexExpr->expression(), {} );
        }

        processAssignment( loc, ctx->expression(), currentVarName, currentIndexExpr );
    }
    CATCH_USER_ERROR

    mlir::Value ParseListener::indexTypeCast( mlir::Location loc, mlir::Value val )
    {
        mlir::IndexType indexTy = builder.getIndexType();
        mlir::Type valTy = val.getType();

        if ( valTy == indexTy )
        {
            return val;
        }

        if ( !valTy.isSignlessInteger( 64 ) && valTy.isInteger() )
        {
            val = castOpIfRequired( loc, val, tyI64 );
            valTy = tyI64;
        }

        // Only support i64, or castable to i64, for now
        if ( !valTy.isSignlessInteger( 64 ) )
        {
            // If it's a non-i64 IntegerType, we could cast up to i64, and then cast that to index.
            throw ExceptionWithContext(
                __FILE__, __LINE__, __func__,
                std::format( "{}internal error: NYI: indexTypeCast from type {} is not supported.\n",
                             formatLocation( loc ), mlirTypeToString( valTy ) ) );
        }

        return builder.create<mlir::arith::IndexCastOp>( loc, indexTy, val );
    }

    mlir::Value ParseListener::parseOr( antlr4::ParserRuleContext *ctx, mlir::Type ty )
    {
        assert( ctx );

        if ( SillyParser::OrExprContext *orCtx = dynamic_cast<SillyParser::OrExprContext *>( ctx ) )
        {
            mlir::Location loc = getStartLocation( ctx );

            // We have at least one OR
            std::vector<SillyParser::BinaryExpressionOrContext *> orOperands = orCtx->binaryExpressionOr();

            // First operand (no special case needed)
            mlir::Value value = parseXor( orOperands[0], ty );
            for ( size_t i = 1; i < orOperands.size(); ++i )
            {
                mlir::Value rhs = parseXor( orOperands[i], ty );

                mlir::Type ty = biggestTypeOf( value.getType(), rhs.getType() );

                value = builder.create<silly::OrOp>( loc, ty, value, rhs ).getResult();
            }

            return value;
        }

        // No OR: just descend to the next level (XOR / single-term)
        return parseXor( ctx, ty );
    }

    mlir::Value ParseListener::parseXor( antlr4::ParserRuleContext *ctx, mlir::Type ty )
    {
        assert( ctx );

        if ( SillyParser::XorExprContext *xorCtx = dynamic_cast<SillyParser::XorExprContext *>( ctx ) )
        {
            mlir::Location loc = getStartLocation( ctx );

            // Has XOR operator(s)
            std::vector<SillyParser::BinaryExpressionXorContext *> xorOperands = xorCtx->binaryExpressionXor();

            mlir::Value value = parseAnd( xorOperands[0], ty );

            for ( size_t i = 1; i < xorOperands.size(); ++i )
            {
                mlir::Value rhs = parseAnd( xorOperands[i], ty );

                mlir::Type ty = biggestTypeOf( value.getType(), rhs.getType() );

                value = builder.create<silly::XorOp>( loc, ty, value, rhs ).getResult();
            }

            return value;
        }

        // No XOR: descend to AND level
        return parseAnd( ctx, ty );
    }

    mlir::Value ParseListener::parseAnd( antlr4::ParserRuleContext *ctx, mlir::Type ty)
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
            mlir::Value value = parseEquality( andOperands[0], ty );

            // Fold the remaining ANDs (left associative)
            for ( size_t i = 1; i < andOperands.size(); ++i )
            {
                mlir::Value rhs = parseEquality( andOperands[i], ty );

                mlir::Type ty = biggestTypeOf( value.getType(), rhs.getType() );

                value = builder.create<silly::AndOp>( loc, ty, value, rhs ).getResult();
            }

            return value;
        }

        // No AND operator: descend directly to the next level (equality)
        return parseEquality( ctx, ty );
    }

    mlir::Value ParseListener::parseEquality( antlr4::ParserRuleContext *ctx, mlir::Type ty )
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
            value = parseComparison( operands[0], ty );

            assert( operands.size() <= 2 );

            if ( operands.size() == 2 )
            {
                mlir::Value rhs = parseComparison( operands[1], ty );

                if ( eqNeCtx->equalityOperator()->EQUALITY_TOKEN() )
                {
                    value = builder.create<silly::EqualOp>( loc, tyI1, value, rhs ).getResult();
                }
                else if ( eqNeCtx->equalityOperator()->NOTEQUAL_TOKEN() )
                {
                    value = builder.create<silly::NotEqualOp>( loc, tyI1, value, rhs ).getResult();
                }
                else
                {
                    throw ExceptionWithContext( __FILE__, __LINE__, __func__,
                                                std::format( "{}internal error: missing EQ or NE token: {}\n",
                                                             formatLocation( loc ), ctx->getText() ) );
                }
            }

            return value;
        }

        // No EQ or NE operators: descend directly to comparison level
        return parseComparison( ctx, ty );
    }

    mlir::Value ParseListener::parseComparison( antlr4::ParserRuleContext *ctx, mlir::Type ty )
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
            value = parseAdditive( operands[0], ty );

            if ( operands.size() == 2 )
            {
                mlir::Value rhs = parseAdditive( operands[1], ty );

                SillyParser::RelationalOperatorContext *op = compareCtx->relationalOperator();
                assert( op );

                if ( op->LESSTHAN_TOKEN() )
                {
                    value = builder.create<silly::LessOp>( loc, tyI1, value, rhs ).getResult();
                }
                else if ( op->GREATERTHAN_TOKEN() )
                {
                    value = builder.create<silly::LessOp>( loc, tyI1, rhs, value ).getResult();
                }
                else if ( op->LESSEQUAL_TOKEN() )
                {
                    value = builder.create<silly::LessEqualOp>( loc, tyI1, value, rhs ).getResult();
                }
                else if ( op->GREATEREQUAL_TOKEN() )
                {
                    value = builder.create<silly::LessEqualOp>( loc, tyI1, rhs, value ).getResult();
                }
                else
                {
                    throw ExceptionWithContext( __FILE__, __LINE__, __func__,
                                                std::format( "{}internal error: missing comparison operator: {}\n",
                                                             formatLocation( loc ), ctx->getText() ) );
                }
            }

            return value;
        }

        // No comparison operators : descend directly to additive level
        return parseAdditive( ctx, ty );
    }

    mlir::Value ParseListener::parseAdditive( antlr4::ParserRuleContext *ctx, mlir::Type ty )
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
            value = parseMultiplicative( operands[0], ty );

            // Fold the remaining additions/subtractions left-associatively
            for ( size_t i = 1; i < numOperands; ++i )
            {
                mlir::Value rhs = parseMultiplicative( operands[i], ty );

                mlir::Type ty = biggestTypeOf( value.getType(), rhs.getType() );

                SillyParser::AdditionOperatorContext *op = ops[i - 1];
                if ( op->PLUSCHAR_TOKEN() )
                {
                    value = builder.create<silly::AddOp>( loc, ty, value, rhs ).getResult();
                }
                else if ( op->MINUS_TOKEN() )
                {
                    value = builder.create<silly::SubOp>( loc, ty, value, rhs ).getResult();
                }
                else
                {
                    throw ExceptionWithContext(
                        __FILE__, __LINE__, __func__,
                        std::format( "{}internal error: missing + or - operator at index {}: {}\n",
                                     formatLocation( loc ), i - 1, ctx->getText() ) );
                }
            }

            return value;
        }
        // Descend directly to multiplicative level if no +-
        return parseMultiplicative( ctx, ty );
    }

    mlir::Value ParseListener::parseMultiplicative( antlr4::ParserRuleContext *ctx, mlir::Type ty )
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
            value = parseUnary( operands[0], ty );

            // Fold the remaining multiplications/divisions left-associatively
            for ( size_t i = 1; i < numOperands; ++i )
            {
                mlir::Value rhs = parseUnary( operands[i], ty );

                mlir::Type ty = biggestTypeOf( value.getType(), rhs.getType() );

                SillyParser::MultiplicativeOperatorContext *op = ops[i - 1];

                if ( op->TIMES_TOKEN() )
                {
                    value = builder.create<silly::MulOp>( loc, ty, value, rhs ).getResult();
                }
                else if ( op->DIV_TOKEN() )
                {
                    value = builder.create<silly::DivOp>( loc, ty, value, rhs ).getResult();
                }
                else
                {
                    throw ExceptionWithContext( __FILE__, __LINE__, __func__,
                                                std::format( "{}internal error: missing * or / operator at index {}\n",
                                                             formatLocation( loc ), i - 1 ) );
                }
            }

            return value;
        }

        // Descend directly to unary level if no * or /
        return parseUnary( ctx, ty );
    }

    mlir::Value ParseListener::parseUnary( antlr4::ParserRuleContext *ctx, mlir::Type ty )
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
            value = parseUnary( unaryOpCtx->unaryExpression(), ty );

            std::string opText = unaryOp->getText();

            if ( opText == "-" )
            {
                // Negation
                value = builder.create<silly::NegOp>( loc, value.getType(), value ).getResult();
            }
            else if ( opText == "+" )
            {
                // Unary plus: identity (no-op)
            }
            else if ( opText == "NOT" )
            {
                if ( !value.getType().isInteger() )
                {
                    throw UserError( loc, std::format( "NOT on non-integer type: {}\n", ctx->getText() ) );
                }

                // NOT x: (x == 0)
                mlir::Value zero =
                    builder.create<mlir::arith::ConstantIntOp>( loc, 0, value.getType().getIntOrFloatBitWidth() );
                value = builder.create<silly::EqualOp>( loc, tyI1, value, zero ).getResult();
            }
            else
            {
                throw ExceptionWithContext(
                    __FILE__, __LINE__, __func__,
                    std::format( "{}internal error: unknown unary operator: {}\n", formatLocation( loc ), opText ) );
            }
        }
        else if ( SillyParser::PrimaryContext *primaryCtx = dynamic_cast<SillyParser::PrimaryContext *>( ctx ) )
        {
            // Case 2: no unary operator, just a primary (# primary)
            value = parsePrimary( primaryCtx->primaryExpression(), ty );
        }
        else
        {
            throw ExceptionWithContext(
                __FILE__, __LINE__, __func__,
                std::format( "{}internal error: unknown unary context: {}\n", formatLocation( loc ), ctx->getText() ) );
        }

        assert( value );

        LLVM_DEBUG( {
            llvm::errs() << "parseUnary: " << ctx->getText() << " -> type ";
            value.getType().dump();
            llvm::errs() << "\n";
        } );

        return value;
    }

    mlir::Value ParseListener::parsePrimary( antlr4::ParserRuleContext *ctx, mlir::Type ty )
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
                value = parseBoolean( loc, booleanNode->getText() );
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

                value = parseInteger( loc, width, integerNode->getText() );
            }
            else if ( tNode *floatNode = lit->FLOAT_PATTERN() )
            {
                mlir::FloatType fty{};
                if ( ty )
                {
                    fty = mlir::dyn_cast<mlir::FloatType>( ty );
                }

                if (!fty)
                {
                   fty = tyF64;
                }

                value = parseFloat( loc, fty, floatNode->getText() );
            }
            else if ( tNode *stringNode = lit->STRING_PATTERN() )
            {
                std::string s = stripQuotes( loc, stringNode->getText() );

                mlir::StringAttr strAttr = builder.getStringAttr( s );
                silly::StringLiteralOp stringLiteral = builder.create<silly::StringLiteralOp>( loc, tyPtr, strAttr );

                value = stringLiteral.getResult();
            }
            else
            {
                throw ExceptionWithContext( __FILE__, __LINE__, __func__,
                                            std::format( "{}internal error: unknown literal type in primary: {}.\n",
                                                         formatLocation( loc ), ctx->getText() ) );
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

            mlir::Value p = searchForInduction( varName );
            if ( p )
            {
                value = p;
            }
            else
            {
                silly::DeclareOp declareOp = lookupDeclareForVar( loc, varName );

                mlir::Type varType = declareOp.getTypeAttr().getValue();

                mlir::SymbolRefAttr symRef = mlir::SymbolRefAttr::get( &dialect.context, varName );

                if ( SillyParser::IndexExpressionContext *indexExpr = scalarOrArrayElement->indexExpression() )
                {
                    value = parseExpression( indexExpr->expression(), {} );

                    mlir::Location iloc = getStartLocation( indexExpr->expression() );
                    mlir::Value i = indexTypeCast( iloc, value );

                    value = builder.create<silly::LoadOp>( loc, varType, symRef, i );
                }
                else
                {
                    if ( declareOp.getSizeAttr() )
                    {
                        if ( mlir::IntegerType ity = mlir::cast<mlir::IntegerType>( varType ) )
                        {
                            unsigned w = ity.getWidth();
                            if ( w == 8 )
                            {
                                varType = tyPtr;
                            }
                        }
                    }

                    value = builder.create<silly::LoadOp>( loc, varType, symRef, mlir::Value{} );
                }
            }
        }
        else if ( SillyParser::CallPrimaryContext *callCtx = dynamic_cast<SillyParser::CallPrimaryContext *>( ctx ) )
        {
            // Function call (# callPrimary)
            value = handleCall( callCtx->callExpression() );
        }
        else if ( SillyParser::ParenExprContext *parenCtx = dynamic_cast<SillyParser::ParenExprContext *>( ctx ) )
        {
            // Parenthesized expression (# parenExpr)
            value = parseExpression( parenCtx->expression(), {} );
        }
        else
        {
            throw ExceptionWithContext( __FILE__, __LINE__, __func__,
                                        std::format( "{}internal error: unknown primary expression: {}.\n",
                                                     formatLocation( loc ), ctx->getText() ) );
        }

        assert( value );

        LLVM_DEBUG( {
            llvm::errs() << "parsePrimary: " << ctx->getText() << " -> type ";
            value.getType().dump();
            llvm::errs() << "\n";
        } );

        return value;
    }
}    // namespace silly

// vim: et ts=4 sw=4
