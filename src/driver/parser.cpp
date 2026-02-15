///
/// @file    parser.cpp
/// @author  Peeter Joot <peeterjoot@pm.me>
/// @brief   altlr4 parse tree listener and MLIR builder.
///
#include <llvm/Support/Debug.h>
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

#include "SillyDialect.hpp"
#include "parser.hpp"

/// Implicit function declaration for the body of a silly language program.
#define ENTRY_SYMBOL_NAME "main"

/// --debug- class for the parser
#define DEBUG_TYPE "silly-parser"

namespace silly
{
    //--------------------------------------------------------------------------
    // DialectCtx members
    DialectCtx::DialectCtx()
    {
        context.getOrLoadDialect<silly::SillyDialect>();
        context.getOrLoadDialect<mlir::func::FuncDialect>();
        context.getOrLoadDialect<mlir::arith::ArithDialect>();
        context.getOrLoadDialect<mlir::memref::MemRefDialect>();
        context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
        context.getOrLoadDialect<mlir::scf::SCFDialect>();
    }

    //--------------------------------------------------------------------------
    // DriverState members

    void DriverState::emitInternalError( mlir::Location loc, const char *compilerfile, unsigned compilerline,
                                         const char *compilerfunc, const std::string &message,
                                         const std::string &programFuncName )
    {
        emitUserError( loc, std::format( "{}:{}:{}: {}", compilerfile, compilerline, compilerfunc, message ),
                       programFuncName, true );
    }

    void DriverState::emitUserError( mlir::Location loc, const std::string &message, const std::string &funcName,
                                     bool internal )
    {
        bool inColor = isatty( fileno( stderr ) ) && colorErrors;
        const char *RED = inColor ? "\033[1;31m" : "";
        const char *CYAN = inColor ? "\033[0;36m" : "";
        const char *RESET = inColor ? "\033[0m" : "";

        if ( internal && errorCount )
        {
            errorCount++;
            return;
        }

        static std::string lastFunc{};
        auto fileLoc = mlir::dyn_cast<mlir::FileLineColLoc>( loc );
        if ( !fileLoc )
        {
            llvm::errs() << std::format( "{}{}error: {}{}\n", RED, internal ? "internal " : "", RESET, message );
        }

        std::string filename = fileLoc.getFilename().str();
        unsigned line = fileLoc.getLine();
        unsigned col = fileLoc.getColumn();

        if ( ( funcName != "" ) && ( funcName != ENTRY_SYMBOL_NAME ) && ( funcName != lastFunc ) )
        {
            llvm::errs() << std::format( "{}: In function ‘{}’:\n", filename, funcName );
        }
        lastFunc = funcName;

        // Print: filename:line:col: error: message
        llvm::errs() << std::format( "{}{}:{}:{}: {}{}error: {}{}\n", CYAN, filename, line, col, RED,
                                     internal ? "internal " : "", RESET, message );

        // Try to read and display the source line
        if ( !filename.empty() || !filename.empty() )
        {
            std::string path = filename.empty() ? filename : filename;

            if ( std::ifstream file{ path } )
            {
                std::string currentLine;
                unsigned currentLineNum = 0;

                while ( std::getline( file, currentLine ) )
                {
                    currentLineNum++;
                    if ( currentLineNum == line )
                    {
                        /* Example output:
                            5 | FOR(INT32 i:(0,2)){PRINT i;}
                              |     ^
                         */
                        // `{:>{}}` - the `^` character, right-aligned to `col` width
                        llvm::errs() << std::format(
                            "{0:5} | {1}\n"
                            "      | {2:>{3}}\n",
                            line, currentLine, "^", col );
                        break;
                    }
                }
            }
        }

        errorCount++;
    }

    //--------------------------------------------------------------------------
    // MlirTypeCache members

    void MlirTypeCache::initialize( mlir::OpBuilder &builder, mlir::MLIRContext *ctx )
    {
        i1 = builder.getI1Type();
        i8 = builder.getI8Type();
        i16 = builder.getI16Type();
        i32 = builder.getI32Type();
        i64 = builder.getI64Type();

        f32 = builder.getF32Type();
        f64 = builder.getF64Type();

        voidT = mlir::LLVM::LLVMVoidType::get( ctx );
        ptr = mlir::LLVM::LLVMPointerType::get( ctx );
    }

    //--------------------------------------------------------------------------
    // PerFunctionState members
    PerFunctionState::PerFunctionState()
        : lastDeclareOp{}, op{}, inductionVariables{}, parameters{}, variables{}, insertionPointStack{}
    {
    }

    inline mlir::Value PerFunctionState::searchForInduction( const std::string &varName )
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

    inline void PerFunctionState::pushInductionVariable( const std::string &varName, mlir::Value i )
    {
        inductionVariables.emplace_back( varName, i );
    }

    inline void PerFunctionState::popInductionVariable()
    {
        inductionVariables.pop_back();
    }

    inline mlir::Value PerFunctionState::searchForParameter( const std::string &varName )
    {
        auto it = parameters.find( varName );
        return ( it != parameters.end() ) ? it->second : nullptr;
    }

    inline mlir::Value PerFunctionState::searchForVariable( const std::string &varName )
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

    inline void PerFunctionState::recordParameterValue( const std::string &varName, mlir::Value i )
    {
        parameters[varName] = i;
    }

    inline void PerFunctionState::recordVariableValue( const std::string &varName, mlir::Value i )
    {
        if ( variables.size() == 0 )
        {
            variables.push_back( {} );
        }

        variables.back()[varName] = i;
    }

    inline void PerFunctionState::startScope()
    {
        variables.push_back( {} );
    }

    inline void PerFunctionState::endScope()
    {
        if ( variables.size() )
        {
            variables.pop_back();
        }
    }

    //--------------------------------------------------------------------------
    // non-member function helpers

    /// A string representation of an mlir::Type
    inline std::string mlirTypeToString( mlir::Type t )
    {
        std::string s;
        llvm::raw_string_ostream( s ) << t;
        return s;
    }

    inline std::string formatLocation( mlir::Location loc )
    {
        if ( mlir::FileLineColLoc fileLoc = mlir::dyn_cast<mlir::FileLineColLoc>( loc ) )
        {
            return std::format( "{}:{}:{}: ", fileLoc.getFilename().str(), fileLoc.getLine(), fileLoc.getColumn() );
        }
        return "";
    }

    mlir::Type biggestTypeOf( mlir::Type ty1, mlir::Type ty2 )
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
    //--------------------------------------------------------------------------
    // ParseListener members
    //
    ParseListener::ParseListener( DriverState &ds, mlir::MLIRContext *context )
        : driverState{ ds },
          ctx{ context },
          builder( ctx ),
          mod( mlir::ModuleOp::create( getStartLocation( nullptr ) ) ),
          mainIP{},
          currentFuncName{},
          functionStateMap{}
    {
        builder.setInsertionPointToStart( mod.getBody() );
        typ.initialize( builder, ctx );
    }

    inline PerFunctionState &ParseListener::funcState( const std::string &funcName )
    {
        if ( !functionStateMap.contains( funcName ) )
        {
            functionStateMap[funcName] = std::make_unique<PerFunctionState>();
        }

        auto &p = functionStateMap[funcName];

        return *p;
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
            driverState.emitInternalError( loc, __FILE__, __LINE__, __func__,
                                           std::format( "boolean value neither TRUE nor FALSE: {}", s ),
                                           currentFuncName );
            return mlir::Value{};
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
        if ( ty == typ.f32 )
        {
            float val = std::stof( s );

            llvm::APFloat apVal( val );

            return builder.create<mlir::arith::ConstantFloatOp>( loc, typ.f32, apVal );
        }
        else
        {
            double val = std::stod( s );

            llvm::APFloat apVal( val );

            return builder.create<mlir::arith::ConstantFloatOp>( loc, typ.f64, apVal );
        }
    }

    silly::StringLiteralOp ParseListener::buildStringLiteral( mlir::Location loc, const std::string &input )
    {
        silly::StringLiteralOp stringLiteral{};

        if ( ( input.size() < 2 ) || ( input.front() != '"' ) || ( input.back() != '"' ) )
        {
            driverState.emitInternalError(
                loc, __FILE__, __LINE__, __func__,
                std::format( "String '{}' was not double quotes enclosed as expected.", input ), currentFuncName );
            return stringLiteral;
        }

        std::string s = input.substr( 1, input.size() - 2 );

        mlir::StringAttr strAttr = builder.getStringAttr( s );

        stringLiteral = builder.create<silly::StringLiteralOp>( loc, typ.ptr, strAttr );

        return stringLiteral;
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
            mlir::FileLineColLoc::get( builder.getStringAttr( driverState.filename ), startLine, startCol + 1 );
        mlir::FileLineColLoc endLoc =
            mlir::FileLineColLoc::get( builder.getStringAttr( driverState.filename ), endLine, endCol + 1 );

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

    inline mlir::Location ParseListener::getTokenLocation( antlr4::Token *token )
    {
        assert( token );
        size_t line = token->getLine();
        size_t column = token->getCharPositionInLine() + 1;

        return mlir::FileLineColLoc::get( builder.getStringAttr( driverState.filename ), line, column );
    }

    inline mlir::Location ParseListener::getTerminalLocation( antlr4::tree::TerminalNode *node )
    {
        assert( node );
        antlr4::Token *token = node->getSymbol();

        return getTokenLocation( token );
    }

    bool ParseListener::isVariableDeclared( const std::string &varName )
    {
        // Get the single scope
        PerFunctionState &f = funcState( currentFuncName );

        mlir::Value v = f.searchForVariable( varName );
        return ( v != nullptr ) ? true : false;
    }

    inline mlir::Value ParseListener::parseExpression( SillyParser::ExpressionContext *ctx, mlir::Type ty )
    {
        mlir::Location loc = getStartLocation( ctx );
        mlir::Value value{};

        if ( !ctx )
        {
            driverState.emitInternalError( loc, __FILE__, __LINE__, __func__, "no ExpressionContext", currentFuncName );
            return value;
        }

        value = parseLowest( ctx, ty );
        if ( !value )
        {
            driverState.emitInternalError( loc, __FILE__, __LINE__, __func__, "parseLowest failed", currentFuncName );
            return value;
        }

        if ( ty )
        {
            value = castOpIfRequired( loc, value, ty );
        }

        return value;
    }

    void ParseListener::registerDeclaration( mlir::Location loc, const std::string &varName, mlir::Type ty,
                                             SillyParser::ArrayBoundsExpressionContext *arrayBounds,
                                             SillyParser::ExpressionContext *assignmentExpression,
                                             const std::vector<SillyParser::ExpressionContext *> *pExpressions )
    {
        int64_t arraySize{};
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

        mlir::Value v = f.searchForVariable( varName );
        if ( v )
        {
            // coverage: error_redeclare.silly
            driverState.emitUserError( loc, std::format( "Variable {} already declared", varName ), currentFuncName );
            return;
        }

        if ( mlir::Operation *op = f.getLastDeclared() )
        {
            builder.setInsertionPointAfter( op );
        }
        else
        {
            mlir::func::FuncOp funcOp = f.getFuncOp();

            mlir::Region &funcRegion = funcOp.getBody();
            if (funcRegion.empty())
            {
                driverState.emitInternalError( loc, __FILE__, __LINE__, __func__,
                                               "Function has empty body", currentFuncName );
                return;
            }

            mlir::Block *entryBlock = &funcRegion.front();

            // Insert declarations at the beginning of the entry block
            // (all DeclareOps should appear before any scf.if/scf.for)
            builder.setInsertionPointToStart( entryBlock );
        }

        std::vector<mlir::Value> initializers;

        if ( pExpressions )
        {
            mlir::Value fill{};

            for ( SillyParser::ExpressionContext *e : *pExpressions )
            {
                mlir::Value init = parseExpression( e, ty );
                if ( !init )
                {
                    driverState.emitInternalError( loc, __FILE__, __LINE__, __func__, "parseExpression failed",
                                                   currentFuncName );
                    return;
                }

                initializers.push_back( init );
            }

            ssize_t remaining = numElements - initializers.size();

            if ( remaining )
            {
                if ( ty == typ.i1 )
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
                    driverState.emitInternalError( loc, __FILE__, __LINE__, __func__, "unknown scalar type.",
                                                   currentFuncName );
                    return;
                }
            }

            if ( initializers.size() > numElements )
            {
                // coverage: error_array_too_many_init.silly, error_init_list1.silly, error_init_list2.silly
                driverState.emitUserError(
                    loc,
                    std::format( "For variable '{}', more initializers ({}) specified than number of elements ({}).\n",
                                 varName, initializers.size(), numElements ),
                    currentFuncName );
                return;
            }

            for ( ssize_t i = 0; i < remaining; i++ )
            {
                assert( fill );
                initializers.push_back( fill );
            }
        }

        mlir::DenseI64ArrayAttr shapeAttr;
        if ( arraySize )
        {
            shapeAttr = builder.getDenseI64ArrayAttr( { arraySize } );
        }
        else
        {
            shapeAttr = builder.getDenseI64ArrayAttr( {} );
        }

        silly::varType varType = builder.getType<silly::varType>( ty, shapeAttr );

        silly::DeclareOp dcl = builder.create<silly::DeclareOp>( loc, varType, initializers );
        f.recordVariableValue( varName, dcl.getResult() );

        f.setLastDeclared( dcl.getOperation() );

        builder.create<silly::DebugNameOp>( loc, dcl.getResult(), varName );

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

            driverState.emitInternalError(
                loc, __FILE__, __LINE__, __func__,
                std::format( "unexpected ExpressionContext alternative: {}", ctx->getText() ), currentFuncName );
            return mlir::Value{};
        }

        SillyParser::BinaryExpressionLowestContext *lowest = expr->binaryExpressionLowest();

        return parseOr( lowest, ty );
    }

    void ParseListener::syntaxError( antlr4::Recognizer *recognizer, antlr4::Token *offendingSymbol, size_t line,
                                     size_t charPositionInLine, const std::string &msg, std::exception_ptr e )
    {
        if ( offendingSymbol )
        {
            mlir::Location loc = getTokenLocation( offendingSymbol );

            driverState.emitUserError( loc, std::format( "parse error: {}", msg ), currentFuncName );
        }
        else
        {
            mlir::Location loc = builder.getUnknownLoc();
            driverState.emitInternalError(
                loc, __FILE__, __LINE__, __func__,
                std::format( "parse error in {}:{}:{}: {}", driverState.filename, line, charPositionInLine, msg ),
                currentFuncName );
        }
    }

    silly::DeclareOp ParseListener::lookupDeclareForVar( mlir::Location loc, const std::string &varName )
    {
        silly::DeclareOp declareOp{};
        PerFunctionState &f = funcState( currentFuncName );

        mlir::Value var = f.searchForVariable( varName );
        if ( !var )
        {
            // coverage: error_induction_var_in_step.silly
            driverState.emitUserError( loc, std::format( "Undeclared variable {}", varName ), currentFuncName );
            return declareOp;
        }

        declareOp = var.getDefiningOp<silly::DeclareOp>();
        assert( declareOp );    // not sure I could trigger NULL declareOp with user code.

        return declareOp;
    }

    mlir::Type ParseListener::parseScalarType( const std::string &ty )
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

    void ParseListener::createScope( mlir::Location startLoc, mlir::Location endLoc, mlir::func::FuncOp funcOp,
                                     const std::string &funcName, const std::vector<std::string> &paramNames )
    {
        LLVM_DEBUG( {
            llvm::errs() << std::format( "createScope: {}: startLoc: {}, endLoc: {}\n", funcName,
                                         formatLocation( startLoc ), formatLocation( endLoc ) );
        } );
        mlir::Block &block = *funcOp.addEntryBlock();
        builder.setInsertionPointToStart( &block );

        PerFunctionState &f = funcState( funcName );

        for ( size_t i = 0; i < funcOp.getNumArguments() && i < paramNames.size(); ++i )
        {
            LLVM_DEBUG( {
                llvm::errs() << std::format( "function {}: parameter{}:\n", funcName, i );
                funcOp.getArgument( i ).dump();
            } );

            mlir::Value param = funcOp.getArgument( i );
            builder.create<silly::DebugNameOp>( startLoc, param, paramNames[i] );
            f.recordParameterValue( paramNames[i], param );
        }

        currentFuncName = funcName;
        f.setFuncOp( funcOp );
    }

    void ParseListener::enterStartRule( SillyParser::StartRuleContext *ctx )
    {
        assert( ctx );
        currentFuncName = ENTRY_SYMBOL_NAME;

        LocPairs locs = getLocations( ctx );

        mlir::FunctionType funcType = builder.getFunctionType( {}, typ.i32 );
        mlir::func::FuncOp funcOp = builder.create<mlir::func::FuncOp>( locs.first, ENTRY_SYMBOL_NAME, funcType );

        std::vector<std::string> paramNames;
        createScope( locs.first, locs.second, funcOp, ENTRY_SYMBOL_NAME, paramNames );
    }


    void ParseListener::enterScopedStatements( SillyParser::ScopedStatementsContext *ctx )
    {
        PerFunctionState &f = funcState( currentFuncName );
        f.startScope();
    }

    void ParseListener::exitScopedStatements( SillyParser::ScopedStatementsContext *ctx )
    {
        PerFunctionState &f = funcState( currentFuncName );
        f.endScope();
    }

    void ParseListener::processReturnLike( mlir::Location loc, SillyParser::ExpressionContext *expression )
    {
        mlir::Type returnType{};
        mlir::Value value{};

        if ( currentFuncName == ENTRY_SYMBOL_NAME )
        {
            returnType = typ.i32;
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
                // coverage: error_return_expr_no_return_type.silly
                driverState.emitUserError(
                    loc,
                    std::format( "return expression found '{}', but no return type for function {}",
                                 expression->getText(), currentFuncName ),
                    currentFuncName );
                return;
            }

            value = parseExpression( expression, returnType );
            if ( !value )
            {
                driverState.emitInternalError( loc, __FILE__, __LINE__, __func__, "parseExpression failed",
                                               currentFuncName );
                return;
            }
        }
        else if ( currentFuncName == ENTRY_SYMBOL_NAME )
        {
            value = builder.create<mlir::arith::ConstantIntOp>( loc, 0, 32 );
        }

        // Create ReturnOp with user specified value:
        if ( value )
        {
            builder.create<mlir::func::ReturnOp>( loc, mlir::ValueRange{ value } );
        }
        else
        {
            builder.create<mlir::func::ReturnOp>( loc, mlir::ValueRange{} );
        }
    }

    void ParseListener::exitStartRule( SillyParser::StartRuleContext *ctx )
    {
        assert( ctx );
        mlir::Location loc = getStartLocation( ctx );

        assert( currentFuncName == ENTRY_SYMBOL_NAME );

        if ( !ctx->exitStatement() )
        {
            processReturnLike( loc, nullptr );
        }

#if 0
        LLVM_DEBUG( {
            llvm::errs() << "exitStartRule done: module dump:\n";
            mod->dump();
        } );
#endif
    }

    void ParseListener::enterFunctionStatement( SillyParser::FunctionStatementContext *ctx )
    {
        assert( ctx );
        LocPairs locs = getLocations( ctx );

        LLVM_DEBUG( {
            llvm::errs() << std::format( "enterFunctionStatement: startLoc: {}, endLoc: {}:\n",
                                         formatLocation( locs.first ), formatLocation( locs.second ) );
        } );

        if ( currentFuncName != ENTRY_SYMBOL_NAME )
        {
            // coverage: error_nested.silly
            //
            // To support this, exitFor would have to pop an insertion point and current-function-name,
            // and we'd have to push an insertion-point/function-name instead of just assuming that
            // we started in main and will return to there.
            driverState.emitUserError( locs.first, std::format( "Nested functions are not currently supported." ),
                                       currentFuncName );
            return;
        }

        mainIP = builder.saveInsertionPoint();

        builder.setInsertionPointToStart( mod.getBody() );

        assert( ctx->IDENTIFIER() );
        std::string funcName = ctx->IDENTIFIER()->getText();

        std::vector<mlir::Type> returns;
        if ( SillyParser::ScalarTypeContext *rt = ctx->scalarType() )
        {
            mlir::Type returnType = parseScalarType( rt->getText() );

            if ( returnType )
            {
                returns.push_back( returnType );
            }
            else
            {
                mlir::Location loc = getStartLocation( ctx );
                driverState.emitInternalError( loc, __FILE__, __LINE__, __func__, "no returnType", currentFuncName );
                return;
            }
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

    void ParseListener::exitFunctionStatement( SillyParser::FunctionStatementContext *ctx )
    {
        assert( ctx );
        builder.restoreInsertionPoint( mainIP );

        currentFuncName = ENTRY_SYMBOL_NAME;
    }

    mlir::Value ParseListener::handleCall( SillyParser::CallExpressionContext *ctx )
    {
        mlir::Value ret{};
        assert( ctx );
        mlir::Location loc = getStartLocation( ctx );
        tNode *id = ctx->IDENTIFIER();
        assert( id );
        std::string funcName = id->getText();
        PerFunctionState &f = funcState( funcName );
        mlir::func::FuncOp funcOp = f.getFuncOp();
        if ( !funcOp )
        {
            driverState.emitInternalError( loc, __FILE__, __LINE__, __func__,
                                           std::format( "no FuncOp found for {}", funcName ), currentFuncName );
            return ret;
        }

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
                // LLVM_DEBUG( { llvm::errs() << std::format( "CALL function {}: param: {}\n", funcName, paramText ) }
                // );

                mlir::Type ty = funcType.getInputs()[i];
                mlir::Value value = parseExpression( p, ty );
                if ( !value )
                {
                    driverState.emitInternalError( loc, __FILE__, __LINE__, __func__, "parseExpression failed",
                                                   currentFuncName );
                    return value;
                }

                parameters.push_back( value );
                i++;
            }
        }

        mlir::TypeRange resultTypes = funcType.getResults();

        silly::CallOp callOp = builder.create<silly::CallOp>( loc, resultTypes, funcName, parameters );

        // Return the first result (or null for void calls)
        if ( !resultTypes.empty() )
        {
            ret = callOp.getResults()[0];
        }

        return ret;
    }

    void ParseListener::enterCallStatement( SillyParser::CallStatementContext *ctx )
    {
        assert( ctx );
        handleCall( ctx->callExpression() );
    }

    void ParseListener::enterDeclareHelper(
        mlir::Location loc, tNode *identifier,
        SillyParser::DeclareAssignmentExpressionContext *declareAssignmentExpression,
        const std::vector<SillyParser::ExpressionContext *> &expressions, tNode *hasInitList,
        SillyParser::ArrayBoundsExpressionContext *arrayBoundsExpression, mlir::Type ty )
    {
        if ( !identifier )
        {
            driverState.emitInternalError( loc, __FILE__, __LINE__, __func__,
                                           "no identifier for declaration processing", currentFuncName );
            return;
        }
        std::string varName = identifier->getText();

        const std::vector<SillyParser::ExpressionContext *> *pExpressions{};
        SillyParser::ExpressionContext *assignmentExpression{};

        if ( hasInitList || expressions.size() )
        {
            if ( declareAssignmentExpression )
            {
                // TODO: no coverage.
                driverState.emitUserError(
                    loc,
                    std::format(
                        "Declaration cannot have both assignment expression and initialization-list expression." ),
                    currentFuncName );
                return;
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

    void ParseListener::enterBoolDeclareStatement( SillyParser::BoolDeclareStatementContext *ctx )
    {
        assert( ctx );
        mlir::Location loc = getStartLocation( ctx );
        enterDeclareHelper( loc, ctx->IDENTIFIER(), ctx->declareAssignmentExpression(), ctx->expression(),
                            ctx->LEFT_CURLY_BRACKET_TOKEN(), ctx->arrayBoundsExpression(), typ.i1 );
    }

    mlir::Type ParseListener::integerDeclarationType( mlir::Location loc, SillyParser::IntTypeContext *ctx )
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
            driverState.emitInternalError( loc, __FILE__, __LINE__, __func__,
                                           "Unsupported signed integer declaration size.", currentFuncName );
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
            ty = typ.f32;
        }
        else if ( ctx->FLOAT64_TOKEN() )
        {
            ty = typ.f64;
        }
        else
        {
            mlir::Location loc = getStartLocation( ctx );
            driverState.emitInternalError( loc, __FILE__, __LINE__, __func__,
                                           "Unsupported floating point declaration size.", currentFuncName );
            return;
        }

        enterDeclareHelper( loc, ctx->IDENTIFIER(), ctx->declareAssignmentExpression(), ctx->expression(),
                            ctx->LEFT_CURLY_BRACKET_TOKEN(), ctx->arrayBoundsExpression(), ty );
    }

    void ParseListener::enterStringDeclareStatement( SillyParser::StringDeclareStatementContext *ctx )
    {
        assert( ctx );
        mlir::Location loc = getStartLocation( ctx );
        assert( ctx->IDENTIFIER() );
        std::string varName = ctx->IDENTIFIER()->getText();
        SillyParser::ArrayBoundsExpressionContext *arrayBounds = ctx->arrayBoundsExpression();
        assert( arrayBounds );

        registerDeclaration( loc, varName, typ.i8, arrayBounds, nullptr, nullptr );

        if ( tNode *theString = ctx->STRING_PATTERN() )
        {
            silly::DeclareOp declareOp = lookupDeclareForVar( loc, varName );
            mlir::Value var = declareOp.getResult();

            silly::StringLiteralOp stringLiteral = buildStringLiteral( loc, theString->getText() );
            if ( stringLiteral )
            {
                mlir::Value i{};
                builder.create<silly::AssignOp>( loc, var, i, stringLiteral );
            }
        }
    }

    void ParseListener::checkForReturnInScope( SillyParser::ScopedStatementsContext *scope, const char *what )
    {
        assert( scope );

        if ( SillyParser::ReturnStatementContext *ret = scope->returnStatement() )
        {
            mlir::Location rLoc = getStartLocation( ret );
            driverState.emitUserError( rLoc, std::format( "RETURN is not currently allowed in a {}", what ),
                                       currentFuncName );
        }
    }

    void ParseListener::createIf( mlir::Location loc, SillyParser::ExpressionContext *predicate, bool saveIP )
    {
        mlir::Value conditionPredicate = parseExpression( predicate, {} );
        if ( !conditionPredicate )
        {
            driverState.emitInternalError( loc, __FILE__, __LINE__, __func__, "parseExpression failed",
                                           currentFuncName );
            return;
        }

        mlir::scf::IfOp ifOp = builder.create<mlir::scf::IfOp>( loc, conditionPredicate,
                                                                /*withElseRegion=*/true );

        if ( saveIP )
        {
            PerFunctionState &f = funcState( currentFuncName );
            f.pushToInsertionPointStack( ifOp.getOperation() );
        }

        mlir::Block &thenBlock = ifOp.getThenRegion().front();
        builder.setInsertionPointToStart( &thenBlock );
    }

    void ParseListener::enterIfStatement( SillyParser::IfStatementContext *ctx )
    {
        assert( ctx );
        mlir::Location loc = getStartLocation( ctx );

        checkForReturnInScope( ctx->scopedStatements(), "IF block" );

        createIf( loc, ctx->expression(), true );
    }

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
            driverState.emitInternalError( loc, __FILE__, __LINE__, __func__,
                                           "Current insertion point must be inside an scf.if then region",
                                           currentFuncName );
            return;
        }

        // Set the insertion point to the start of the else region's (first) block.
        mlir::Region &elseRegion = ifOp.getElseRegion();
        mlir::Block &elseBlock = elseRegion.front();
        builder.setInsertionPointToStart( &elseBlock );
    }

    void ParseListener::enterElseStatement( SillyParser::ElseStatementContext *ctx )
    {
        assert( ctx );
        mlir::Location loc = getStartLocation( ctx );

        checkForReturnInScope( ctx->scopedStatements(), "ELSE block" );

        selectElseBlock( loc, ctx->getText() );
    }

    void ParseListener::enterElifStatement( SillyParser::ElifStatementContext *ctx )
    {
        assert( ctx );
        mlir::Location loc = getStartLocation( ctx );

        checkForReturnInScope( ctx->scopedStatements(), "ELIF block" );

        selectElseBlock( loc, ctx->getText() );

        createIf( loc, ctx->expression(), false );
    }

    void PerFunctionState::pushToInsertionPointStack( mlir::Operation *op )
    {
        insertionPointStack.push_back( op );
    }

    void PerFunctionState::popFromInsertionPointStack( mlir::OpBuilder &builder )
    {
        builder.setInsertionPointAfter( insertionPointStack.back() );
        insertionPointStack.pop_back();
    }

    bool PerFunctionState::haveInsertionPointStack()
    {
        return ( insertionPointStack.size() != 0 );
    }

    void ParseListener::exitIfElifElseStatement( SillyParser::IfElifElseStatementContext *ctx )
    {
        PerFunctionState &f = funcState( currentFuncName );
        f.popFromInsertionPointStack( builder );
    }

    void ParseListener::enterForStatement( SillyParser::ForStatementContext *ctx )
    {
        assert( ctx );
        mlir::Location loc = getStartLocation( ctx );

        LLVM_DEBUG( { llvm::errs() << std::format( "For: {}\n", ctx->getText() ); } );

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

        bool declared = isVariableDeclared( varName );
        if ( declared )
        {
            // coverage: error_shadow_induction.silly
            driverState.emitUserError( loc,
                                       std::format( "Induction variable {} clashes with declared variable\n", varName ),
                                       currentFuncName );
            return;
        }

        PerFunctionState &f = funcState( currentFuncName );
        mlir::Value p = f.searchForInduction( varName );
        if ( p )
        {
            // coverage: error_triple_nested_for_with_shadowing.silly error_nested_ivar_conflict.silly
            driverState.emitUserError( loc, std::format( "Induction variable {} used by enclosing FOR\n", varName ),
                                       currentFuncName );
            return;
        }

        mlir::Type elemType = integerDeclarationType( loc, ctx->intType() );

        std::string s;
        if ( pStart )
        {
            start = parseExpression( pStart, elemType );
            if ( !start )
            {
                driverState.emitInternalError( loc, __FILE__, __LINE__, __func__, "parseExpression failed",
                                               currentFuncName );
                return;
            }
        }
        else
        {
            driverState.emitInternalError( loc, __FILE__, __LINE__, __func__,
                                           std::format( "FOR loop: expected start index: {}", ctx->getText() ),
                                           currentFuncName );
            return;
        }

        if ( pEnd )
        {
            end = parseExpression( pEnd, elemType );
            if ( !end )
            {
                driverState.emitInternalError( loc, __FILE__, __LINE__, __func__, "parseExpression failed",
                                               currentFuncName );
                return;
            }
        }
        else
        {
            driverState.emitInternalError( loc, __FILE__, __LINE__, __func__,
                                           std::format( "FOR loop: expected end index: {}", ctx->getText() ),
                                           currentFuncName );
            return;
        }

        if ( pStep )
        {
            step = parseExpression( pStep, elemType );
            if ( !step )
            {
                driverState.emitInternalError( loc, __FILE__, __LINE__, __func__, "parseExpression failed",
                                               currentFuncName );
                return;
            }
        }
        else
        {
            mlir::IntegerType ity = mlir::dyn_cast<mlir::IntegerType>( elemType );
            unsigned width = ity.getWidth();

            //'scf.for' op failed to verify that all of {lowerBound, upperBound, step} have same type
            step = builder.create<mlir::arith::ConstantIntOp>( loc, 1, width );
            step = castOpIfRequired( loc, step, elemType );
        }

        checkForReturnInScope( ctx->scopedStatements(), "FOR loop body" );

        mlir::scf::ForOp forOp = builder.create<mlir::scf::ForOp>( loc, start, end, step );
        f.pushToInsertionPointStack( forOp.getOperation() );

        mlir::Block &loopBody = forOp.getRegion().front();
        builder.setInsertionPointToStart( &loopBody );

        mlir::Value inductionVar = loopBody.getArgument( 0 );
        f.pushInductionVariable( varName, inductionVar );

        mlir::Location varLoc = getTerminalLocation( ctx->IDENTIFIER() );
        builder.create<silly::DebugNameOp>( varLoc, inductionVar, varName );
    }

    void ParseListener::exitForStatement( SillyParser::ForStatementContext *ctx )
    {
        assert( ctx );
        PerFunctionState &f = funcState( currentFuncName );
        if ( f.haveInsertionPointStack() )
        {
            f.popFromInsertionPointStack( builder );
            f.popInductionVariable();
        }
        else
        {
            mlir::Location loc = getStartLocation( ctx );
            driverState.emitInternalError( loc, __FILE__, __LINE__, __func__, "empty insertionPointStack",
                                           currentFuncName );
        }
    }

    void ParseListener::handlePrint( mlir::Location loc, const std::vector<SillyParser::ExpressionContext *> &args,
                                     const std::string &errorContextString, PrintFlags pf )
    {
        std::vector<mlir::Value> vargs;
        for ( SillyParser::ExpressionContext *parg : args )
        {
            mlir::Value v = parseExpression( parg, {} );
            if ( !v )
            {
                driverState.emitInternalError( loc, __FILE__, __LINE__, __func__, "parseExpression failed",
                                               currentFuncName );
                return;
            }

            vargs.push_back( v );
        }

        mlir::arith::ConstantIntOp constFlagOp = builder.create<mlir::arith::ConstantIntOp>( loc, pf, 32 );
        builder.create<silly::PrintOp>( loc, constFlagOp, vargs );
    }

    void ParseListener::enterPrintStatement( SillyParser::PrintStatementContext *ctx )
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

    void ParseListener::enterErrorStatement( SillyParser::ErrorStatementContext *ctx )
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

    void ParseListener::enterAbortStatement( SillyParser::AbortStatementContext *ctx )
    {
        assert( ctx );
        mlir::Location loc = getStartLocation( ctx );

        builder.create<silly::AbortOp>( loc );
    }

    void ParseListener::enterGetStatement( SillyParser::GetStatementContext *ctx )
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
            silly::varType varTy = mlir::cast<silly::varType>( declareOp.getVar().getType() );
            mlir::Type elemType = varTy.getElementType();
            mlir::DenseI64ArrayAttr shapeAttr = varTy.getShape();
            llvm::ArrayRef<int64_t> shape = shapeAttr.asArrayRef();

            mlir::Value optIndexValue{};
            if ( SillyParser::IndexExpressionContext *indexExpr = scalarOrArrayElement->indexExpression() )
            {
                mlir::Value indexValue = parseExpression( indexExpr->expression(), {} );
                if ( !indexValue )
                {
                    driverState.emitInternalError( loc, __FILE__, __LINE__, __func__, "parseExpression failed",
                                                   currentFuncName );
                    return;
                }

                mlir::Location iloc = getStartLocation( indexExpr->expression() );
                optIndexValue = indexTypeCast( iloc, indexValue );
            }
            else if ( !shape.empty() )
            {
                // TODO: no coverage.
                driverState.emitUserError( loc, std::format( "Attempted GET to string literal or array?" ),
                                           currentFuncName );
                return;
            }
            else
            {
                // Scalar: load the value
            }

            mlir::Value var = declareOp.getResult();

            silly::GetOp resultValue = builder.create<silly::GetOp>( loc, elemType );
            builder.create<silly::AssignOp>( loc, var, optIndexValue, resultValue );
        }
        else
        {
            driverState.emitInternalError( loc, __FILE__, __LINE__, __func__,
                                           std::format( "unexpected get context {}", ctx->getText() ),
                                           currentFuncName );
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
    {
        assert( ctx );
        mlir::Location loc = getStartLocation( ctx );

        processReturnLike( loc, ctx->expression() );
    }

    void ParseListener::enterExitStatement( SillyParser::ExitStatementContext *ctx )
    {
        assert( ctx );
        mlir::Location loc = getStartLocation( ctx );

        processReturnLike( loc, ctx->expression() );
    }

    void ParseListener::processAssignment( mlir::Location loc, SillyParser::ExpressionContext *exprContext,
                                           const std::string &currentVarName, mlir::Value currentIndexExpr )
    {
        mlir::Value resultValue = parseExpression( exprContext, {} );
        if ( !resultValue )
        {
            mlir::Location loc = getStartLocation( exprContext );
            driverState.emitInternalError( loc, __FILE__, __LINE__, __func__, "no resultValue for expression",
                                           currentFuncName );
            return;
        }

        silly::DeclareOp declareOp = lookupDeclareForVar( loc, currentVarName );
        mlir::Value var = declareOp.getResult();

        assert( resultValue );

        mlir::BlockArgument ba = mlir::dyn_cast<mlir::BlockArgument>( resultValue );
        mlir::Operation *op = resultValue.getDefiningOp();
        mlir::Value i{};

        // Don't check if it's a StringLiteralOp if it's an induction variable, since op will be nullptr
        if ( !ba && isa<silly::StringLiteralOp>( op ) )
        {
            builder.create<silly::AssignOp>( loc, var, i, resultValue );
        }
        else
        {
            if ( currentIndexExpr )
            {
                mlir::Value i = indexTypeCast( loc, currentIndexExpr );

                silly::AssignOp assign = builder.create<silly::AssignOp>( loc, var, i, resultValue );

                LLVM_DEBUG( {
                    mlir::OpPrintingFlags flags;
                    flags.enableDebugInfo( true );

                    assign->print( llvm::outs(), flags );
                    llvm::outs() << "\n";
                } );
            }
            else
            {
                builder.create<silly::AssignOp>( loc, var, i, resultValue );
            }
        }
    }

    void ParseListener::enterAssignmentStatement( SillyParser::AssignmentStatementContext *ctx )
    {
        assert( ctx );
        mlir::Location loc = getStartLocation( ctx );
        SillyParser::ScalarOrArrayElementContext *lhs = ctx->scalarOrArrayElement();
        assert( lhs );
        assert( lhs->IDENTIFIER() );
        std::string currentVarName = lhs->IDENTIFIER()->getText();

        SillyParser::IndexExpressionContext *indexExpr = lhs->indexExpression();
        mlir::Value currentIndexExpr = mlir::Value{};

        bool declared = isVariableDeclared( currentVarName );
        if ( !declared )
        {
            // coverage: error_undeclare.silly
            driverState.emitUserError(
                loc, std::format( "Attempt to assign to undeclared variable: {}\n", currentVarName ), currentFuncName );
            return;
        }

        if ( indexExpr )
        {
            currentIndexExpr = parseExpression( indexExpr->expression(), {} );
            if ( !currentIndexExpr )
            {
                driverState.emitInternalError( loc, __FILE__, __LINE__, __func__, "parseExpression failed",
                                               currentFuncName );
                return;
            }
        }

        processAssignment( loc, ctx->expression(), currentVarName, currentIndexExpr );
    }

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
            val = castOpIfRequired( loc, val, typ.i64 );
            valTy = typ.i64;
        }

        // Only support i64, or castable to i64, for now
        if ( !valTy.isSignlessInteger( 64 ) )
        {
            // If it's a non-i64 IntegerType, we could cast up to i64, and then cast that to index.
            driverState.emitInternalError(
                loc, __FILE__, __LINE__, __func__,
                std::format( "NYI: indexTypeCast from type {} is not supported.", mlirTypeToString( valTy ) ),
                currentFuncName );
            return mlir::Value{};
        }

        return builder.create<mlir::arith::IndexCastOp>( loc, indexTy, val );
    }

    inline mlir::Value ParseListener::createBinaryArith( mlir::Location loc, silly::ArithBinOpKind what, mlir::Type ty,
                                                         mlir::Value lhs, mlir::Value rhs )
    {
        return builder.create<silly::ArithBinOp>( loc, ty, silly::ArithBinOpKindAttr::get( this->ctx, what ), lhs, rhs )
            .getResult();
    }

    inline mlir::Value ParseListener::createBinaryCmp( mlir::Location loc, silly::CmpBinOpKind what, mlir::Value lhs,
                                                       mlir::Value rhs )
    {
        return builder.create<silly::CmpBinOp>( loc, typ.i1, silly::CmpBinOpKindAttr::get( this->ctx, what ), lhs, rhs )
            .getResult();
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
            if ( !value )
            {
                driverState.emitInternalError( loc, __FILE__, __LINE__, __func__, "parseXor failed", currentFuncName );
                return value;
            }

            for ( size_t i = 1; i < orOperands.size(); ++i )
            {
                mlir::Value rhs = parseXor( orOperands[i], ty );
                if ( !rhs )
                {
                    driverState.emitInternalError( loc, __FILE__, __LINE__, __func__, "parseXor failed",
                                                   currentFuncName );
                    return rhs;
                }

                mlir::Type ty = biggestTypeOf( value.getType(), rhs.getType() );

                value = createBinaryArith( loc, silly::ArithBinOpKind::Or, ty, value, rhs );
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
            if ( !value )
            {
                driverState.emitInternalError( loc, __FILE__, __LINE__, __func__, "parseAnd failed", currentFuncName );
                return value;
            }

            for ( size_t i = 1; i < xorOperands.size(); ++i )
            {
                mlir::Value rhs = parseAnd( xorOperands[i], ty );
                if ( !rhs )
                {
                    driverState.emitInternalError( loc, __FILE__, __LINE__, __func__, "parseAnd failed",
                                                   currentFuncName );
                    return rhs;
                }

                mlir::Type ty = biggestTypeOf( value.getType(), rhs.getType() );

                value = createBinaryArith( loc, silly::ArithBinOpKind::Xor, ty, value, rhs );
            }

            return value;
        }

        // No XOR: descend to AND level
        return parseAnd( ctx, ty );
    }

    mlir::Value ParseListener::parseAnd( antlr4::ParserRuleContext *ctx, mlir::Type ty )
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
            if ( !value )
            {
                driverState.emitInternalError( loc, __FILE__, __LINE__, __func__, "parseEquality failed",
                                               currentFuncName );
                return value;
            }

            // Fold the remaining ANDs (left associative)
            for ( size_t i = 1; i < andOperands.size(); ++i )
            {
                mlir::Value rhs = parseEquality( andOperands[i], ty );
                if ( !rhs )
                {
                    driverState.emitInternalError( loc, __FILE__, __LINE__, __func__, "parseEquality failed",
                                                   currentFuncName );
                    return rhs;
                }

                mlir::Type ty = biggestTypeOf( value.getType(), rhs.getType() );

                value = createBinaryArith( loc, silly::ArithBinOpKind::And, ty, value, rhs );
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
            if ( !value )
            {
                driverState.emitInternalError( loc, __FILE__, __LINE__, __func__, "parseComparison failed",
                                               currentFuncName );
                return value;
            }

            assert( operands.size() <= 2 );

            if ( operands.size() == 2 )
            {
                mlir::Value rhs = parseComparison( operands[1], ty );
                if ( !rhs )
                {
                    driverState.emitInternalError( loc, __FILE__, __LINE__, __func__, "parseComparison failed",
                                                   currentFuncName );
                    return rhs;
                }

                if ( eqNeCtx->equalityOperator()->EQUALITY_TOKEN() )
                {
                    value = createBinaryCmp( loc, silly::CmpBinOpKind::Equal, value, rhs );
                }
                else if ( eqNeCtx->equalityOperator()->NOTEQUAL_TOKEN() )
                {
                    value = createBinaryCmp( loc, silly::CmpBinOpKind::NotEqual, value, rhs );
                }
                else
                {
                    driverState.emitInternalError( loc, __FILE__, __LINE__, __func__,
                                                   std::format( "missing EQ or NE token: {}", ctx->getText() ),
                                                   currentFuncName );
                    value = mlir::Value{};
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
            if ( !value )
            {
                driverState.emitInternalError( loc, __FILE__, __LINE__, __func__, "parseAdditive failed",
                                               currentFuncName );
                return value;
            }

            if ( operands.size() == 2 )
            {
                mlir::Value rhs = parseAdditive( operands[1], ty );
                if ( !rhs )
                {
                    driverState.emitInternalError( loc, __FILE__, __LINE__, __func__, "parseAdditive failed",
                                                   currentFuncName );
                    return rhs;
                }

                SillyParser::RelationalOperatorContext *op = compareCtx->relationalOperator();
                assert( op );

                if ( op->LESSTHAN_TOKEN() )
                {
                    value = createBinaryCmp( loc, silly::CmpBinOpKind::Less, value, rhs );
                }
                else if ( op->GREATERTHAN_TOKEN() )
                {
                    value = createBinaryCmp( loc, silly::CmpBinOpKind::Less, rhs, value );
                }
                else if ( op->LESSEQUAL_TOKEN() )
                {
                    value = createBinaryCmp( loc, silly::CmpBinOpKind::LessEq, value, rhs );
                }
                else if ( op->GREATEREQUAL_TOKEN() )
                {
                    value = createBinaryCmp( loc, silly::CmpBinOpKind::LessEq, rhs, value );
                }
                else
                {
                    driverState.emitInternalError( loc, __FILE__, __LINE__, __func__,
                                                   std::format( "missing comparison operator: {}", ctx->getText() ),
                                                   currentFuncName );
                    value = mlir::Value{};
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
            if ( !value )
            {
                driverState.emitInternalError( loc, __FILE__, __LINE__, __func__, "parseMultiplicative failed",
                                               currentFuncName );
                return value;
            }

            // Fold the remaining additions/subtractions left-associatively
            for ( size_t i = 1; i < numOperands; ++i )
            {
                mlir::Value rhs = parseMultiplicative( operands[i], ty );
                if ( !rhs )
                {
                    driverState.emitInternalError( loc, __FILE__, __LINE__, __func__, "parseMultiplicative failed",
                                                   currentFuncName );
                    return rhs;
                }

                mlir::Type ty = biggestTypeOf( value.getType(), rhs.getType() );

                SillyParser::AdditionOperatorContext *op = ops[i - 1];
                if ( op->PLUSCHAR_TOKEN() )
                {
                    value = createBinaryArith( loc, silly::ArithBinOpKind::Add, ty, value, rhs );
                }
                else if ( op->MINUS_TOKEN() )
                {
                    value = createBinaryArith( loc, silly::ArithBinOpKind::Sub, ty, value, rhs );
                }
                else
                {
                    driverState.emitInternalError(
                        loc, __FILE__, __LINE__, __func__,
                        std::format( "missing + or - operator at index {}: {}", i - 1, ctx->getText() ),
                        currentFuncName );
                    break;
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
            if ( !value )
            {
                driverState.emitInternalError( loc, __FILE__, __LINE__, __func__, "parseUnary failed",
                                               currentFuncName );
                return value;
            }

            // Fold the remaining multiplications/divisions left-associatively
            for ( size_t i = 1; i < numOperands; ++i )
            {
                mlir::Value rhs = parseUnary( operands[i], ty );
                if ( !rhs )
                {
                    driverState.emitInternalError( loc, __FILE__, __LINE__, __func__, "parseUnary failed",
                                                   currentFuncName );
                    return rhs;
                }

                mlir::Type ty = biggestTypeOf( value.getType(), rhs.getType() );

                SillyParser::MultiplicativeOperatorContext *op = ops[i - 1];

                if ( op->TIMES_TOKEN() )
                {
                    value = createBinaryArith( loc, silly::ArithBinOpKind::Mul, ty, value, rhs );
                }
                else if ( op->DIV_TOKEN() )
                {
                    value = createBinaryArith( loc, silly::ArithBinOpKind::Div, ty, value, rhs );
                }
                else if ( op->MOD_TOKEN() )
                {
                    value = createBinaryArith( loc, silly::ArithBinOpKind::Mod, ty, value, rhs );
                }
                else
                {
                    driverState.emitInternalError( loc, __FILE__, __LINE__, __func__,
                                                   std::format( "missing * or / operator at index {}", i - 1 ),
                                                   currentFuncName );
                    break;
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
            if ( !value )
            {
                driverState.emitInternalError( loc, __FILE__, __LINE__, __func__, "parseUnary failed",
                                               currentFuncName );
                return value;
            }

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
                    // coverage: error_notfloat.silly
                    driverState.emitUserError( loc, std::format( "NOT on non-integer type\n" ), currentFuncName );
                    return mlir::Value{};
                }

                // NOT x: (x == 0)
                mlir::Value zero =
                    builder.create<mlir::arith::ConstantIntOp>( loc, 0, value.getType().getIntOrFloatBitWidth() );
                value = createBinaryCmp( loc, silly::CmpBinOpKind::Equal, value, zero );
            }
            else
            {
                driverState.emitInternalError( loc, __FILE__, __LINE__, __func__,
                                               std::format( "unknown unary operator: {}", opText ), currentFuncName );
                return value;
            }
        }
        else if ( SillyParser::PrimaryContext *primaryCtx = dynamic_cast<SillyParser::PrimaryContext *>( ctx ) )
        {
            // Case 2: no unary operator, just a primary (# primary)
            value = parsePrimary( primaryCtx->primaryExpression(), ty );
        }
        else
        {
            driverState.emitInternalError( loc, __FILE__, __LINE__, __func__,
                                           std::format( "unknown unary context: {}", ctx->getText() ),
                                           currentFuncName );
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

                if ( !fty )
                {
                    fty = typ.f64;
                }

                value = parseFloat( loc, fty, floatNode->getText() );
            }
            else if ( tNode *stringNode = lit->STRING_PATTERN() )
            {
                silly::StringLiteralOp stringLiteral = buildStringLiteral( loc, stringNode->getText() );

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
                driverState.emitInternalError( loc, __FILE__, __LINE__, __func__,
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

            PerFunctionState &f = funcState( currentFuncName );
            mlir::Value iVar = f.searchForInduction( varName );
            mlir::Value pVar = f.searchForParameter( varName );
            if ( iVar )
            {
                value = iVar;
            }
            else if ( pVar )
            {
                value = pVar;
            }
            else
            {
                silly::DeclareOp declareOp = lookupDeclareForVar( loc, varName );
                if ( !declareOp )
                {
                    driverState.emitInternalError( loc, __FILE__, __LINE__, __func__,
                                                   std::format( "DeclareOp lookup for variable {} failed", varName ),
                                                   currentFuncName );
                    return value;
                }

                mlir::Value var = declareOp.getResult();
                silly::varType varTy = mlir::cast<silly::varType>( declareOp.getVar().getType() );
                mlir::Type elemType = varTy.getElementType();
                mlir::Value i{};

                if ( SillyParser::IndexExpressionContext *indexExpr = scalarOrArrayElement->indexExpression() )
                {
                    value = parseExpression( indexExpr->expression(), {} );
                    if ( !value )
                    {
                        driverState.emitInternalError( loc, __FILE__, __LINE__, __func__, "parseExpression failed",
                                                       currentFuncName );
                        return value;
                    }

                    mlir::Location iloc = getStartLocation( indexExpr->expression() );
                    i = indexTypeCast( iloc, value );

                    value = builder.create<silly::LoadOp>( loc, mlir::TypeRange{ elemType }, var, i );
                }
                else
                {
                    mlir::DenseI64ArrayAttr shapeAttr = varTy.getShape();
                    llvm::ArrayRef<int64_t> shape = shapeAttr.asArrayRef();

                    if ( !shape.empty() )
                    {
                        if ( mlir::IntegerType ity = mlir::cast<mlir::IntegerType>( elemType ) )
                        {
                            unsigned w = ity.getWidth();
                            if ( w == 8 )
                            {
                                elemType = typ.ptr;    // HACK.  Assumes that the only use of INT8[] is for STRING.
                            }
                        }
                    }

                    value = builder.create<silly::LoadOp>( loc, mlir::TypeRange{ elemType }, var, i );
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
            if ( !value )
            {
                driverState.emitInternalError( loc, __FILE__, __LINE__, __func__, "parseExpression failed",
                                               currentFuncName );
                return value;
            }
        }
        else
        {
            driverState.emitInternalError( loc, __FILE__, __LINE__, __func__,
                                           std::format( "unknown primary expression: {}", ctx->getText() ),
                                           currentFuncName );
            return value;
        }

        assert( value );

#if 0
        LLVM_DEBUG( {
            llvm::errs() << "parsePrimary: " << ctx->getText() << " -> type ";
            value.getType().dump();
            llvm::errs() << "\n";
        } );
#endif

        return value;
    }
}    // namespace silly

// vim: et ts=4 sw=4
