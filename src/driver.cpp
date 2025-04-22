#include <antlr4-runtime.h>
#include <assert.h>

#include <format>
#include <fstream>
#include <unordered_map>

#include "ToyBaseListener.h"
#include "ToyDialect.h"
#include "ToyLexer.h"
#include "ToyParser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"

// Define a category for Toy Calculator options
static llvm::cl::OptionCategory ToyCategory( "Toy Calculator Options" );

// Command-line option for input file
static llvm::cl::opt<std::string> inputFilename(
    llvm::cl::Positional, llvm::cl::desc( "<input file>" ),
    llvm::cl::init( "-" ), llvm::cl::value_desc( "filename" ),
    llvm::cl::cat( ToyCategory ), llvm::cl::NotHidden );

enum class semantic_errors : int
{
    not_an_error,
    variable_already_declared,
    variable_not_declared,
    variable_not_assigned,
    unknown_error
};

enum class variable_state : int
{
    undeclared,
    declared,
    assigned
};

class semantic_exception : public std::exception
{
   public:
    semantic_exception()
    {
    }

    const char *what()
    {
        return "semantic error";
    }
};

class DialectCtx
{
   public:
    mlir::MLIRContext context;

    DialectCtx()
    {
        context.getOrLoadDialect<toy::ToyDialect>();
        context.getOrLoadDialect<mlir::arith::ArithDialect>();
        context.getOrLoadDialect<mlir::memref::MemRefDialect>();
    }
};

class MLIRListener : public ToyBaseListener
{
   private:
    DialectCtx dialect;
    mlir::OpBuilder builder;
    mlir::ModuleOp mod;
    std::string filename;
    toy::ProgramOp programOp;
    std::string currentVarName;
    mlir::Location currentAssignLoc;
    semantic_errors lastSemError{ semantic_errors::not_an_error };
    std::unordered_map<std::string, variable_state> var_states;
    std::map<std::string, mlir::Value>
        var_storage;    // Maps variable names to memref<f64>
    bool assignmentTargetValid;

    mlir::Location getLocation( antlr4::ParserRuleContext *ctx )
    {
        if ( ctx )
        {
            size_t line = ctx->getStart()->getLine();
            size_t col = ctx->getStart()->getCharPositionInLine();
            return mlir::FileLineColLoc::get( builder.getStringAttr( filename ),
                                              line, col );
        }
        return mlir::UnknownLoc::get( &dialect.context );
    }

    std::string formatLocation( mlir::Location loc )
    {
        if ( auto fileLoc = mlir::dyn_cast<mlir::FileLineColLoc>( loc ) )
        {
            return std::format( "{}:{}:{}: ", fileLoc.getFilename().str(),
                                fileLoc.getLine(), fileLoc.getColumn() );
        }
        return "";
    }

   public:
    MLIRListener( const std::string &_filename )
        : dialect(),
          builder( &dialect.context ),
          mod( mlir::ModuleOp::create( builder.getUnknownLoc() ) ),
          filename( _filename ),
          currentAssignLoc( builder.getUnknownLoc() )
    {
        builder.setInsertionPointToStart( mod.getBody() );
    }

    mlir::ModuleOp &getModule()
    {
        return mod;
    }

    void enterStartRule( ToyParser::StartRuleContext *ctx ) override
    {
        auto loc = getLocation( ctx );
        programOp = builder.create<toy::ProgramOp>( loc );
        programOp.getBody().push_back( new mlir::Block() );
        builder.setInsertionPointToStart( &programOp.getBody().front() );
    }

    void exitStartRule( ToyParser::StartRuleContext *ctx ) override
    {
        builder.setInsertionPointToEnd( mod.getBody() );
        if ( lastSemError != semantic_errors::not_an_error )
        {
            // don't care what the error was, since we already logged it to the
            // console.  Just throw, avoiding future LLVM IR lowering:
            throw semantic_exception();
        }
    }

    void enterDeclare( ToyParser::DeclareContext *ctx ) override
    {
        auto loc = getLocation( ctx );
        auto varName = ctx->VARIABLENAME()->getText();
        auto varState = var_states[varName];
        if ( varState == variable_state::declared )
        {
            lastSemError = semantic_errors::variable_already_declared;
            llvm::errs() << std::format(
                "{}error: Variable {} already declared in DCL\n",
                formatLocation( loc ), varName );
            return;
        }
        if ( varState == variable_state::undeclared )
        {
            var_states[varName] = variable_state::declared;
            // Allocate memref<f64> for the variable
            auto memrefType = mlir::MemRefType::get( {}, builder.getF64Type() );
            auto allocaOp =
                builder.create<mlir::memref::AllocaOp>( loc, memrefType );
            var_storage[varName] = allocaOp.getResult();
            builder.create<toy::DeclareOp>( loc,
                                            builder.getStringAttr( varName ) );
        }
    }

    void enterPrint( ToyParser::PrintContext *ctx ) override
    {
        auto loc = getLocation( ctx );
        auto varName = ctx->VARIABLENAME()->getText();
        auto varState = var_states[varName];
        if ( varState == variable_state::undeclared )
        {
            lastSemError = semantic_errors::variable_not_declared;
            llvm::errs() << std::format(
                "{}error: Variable {} not declared in PRINT\n",
                formatLocation( loc ), varName );
            return;
        }
        if ( varState != variable_state::assigned )
        {
            lastSemError = semantic_errors::variable_not_assigned;
            llvm::errs() << std::format(
                "{}error: Variable {} not assigned in PRINT\n",
                formatLocation( loc ), varName );
            return;
        }
        builder.create<toy::PrintOp>( loc, builder.getStringAttr( varName ) );
    }

    void enterAssignment( ToyParser::AssignmentContext *ctx ) override
    {
        assignmentTargetValid = true;
        auto loc = getLocation( ctx );
        currentVarName = ctx->VARIABLENAME()->getText();
        auto varState = var_states[currentVarName];
        if ( varState == variable_state::declared )
        {
            var_states[currentVarName] = variable_state::assigned;
        }
        else if ( varState != variable_state::declared )
        {
            lastSemError = semantic_errors::variable_not_declared;
            llvm::errs() << std::format(
                "{}error: Variable {} not declared in assignment\n",
                formatLocation( loc ), currentVarName );
            assignmentTargetValid = false;
        }
        currentAssignLoc = loc;
    }

    void enterUnaryexpression( ToyParser::UnaryexpressionContext *ctx ) override
    {
        if ( !assignmentTargetValid )
        {
            return;
        }
        auto loc = getLocation( ctx );
        auto lhs = ctx->element();
        auto op = ctx->unaryoperator()->getText();

        mlir::Value lhsValue;
        if ( lhs->INTEGERLITERAL() )
        {
            int64_t val = std::stoi( lhs->INTEGERLITERAL()->getText() );
            lhsValue =
                builder.create<mlir::arith::ConstantIntOp>( loc, val, 64 );
        }
        else if ( lhs->VARIABLENAME() )
        {
            auto varName = lhs->VARIABLENAME()->getText();
            auto varState = var_states[varName];
            if ( varState == variable_state::undeclared )
            {
                lastSemError = semantic_errors::variable_not_declared;
                llvm::errs() << std::format(
                    "{}error: Variable {} not declared in unary expr\n",
                    formatLocation( loc ), varName );
                return;
            }
            if ( varState != variable_state::assigned )
            {
                lastSemError = semantic_errors::variable_not_assigned;
                llvm::errs() << std::format(
                    "{}error: Variable {} not assigned in unary expr\n",
                    formatLocation( loc ), varName );
                return;
            }
            // Load from memref<f64>
            auto memref = var_storage[varName];
            lhsValue = builder.create<mlir::memref::LoadOp>( loc, memref );
        }
        else
        {
            lastSemError = semantic_errors::unknown_error;
            llvm::errs() << std::format( "{}error: Invalid unary operand\n",
                                         formatLocation( loc ) );
            return;
        }

        // Create UnaryOp (supports + or -)
        if ( op == "" )
        {
            op = "+";
        }

        auto unaryOp = builder.create<toy::UnaryOp>(
            loc, builder.getF64Type(), builder.getStringAttr( op ), lhsValue );

        if ( !currentVarName.empty() )
        {
            // Store result to memref<f64>
            auto memref = var_storage[currentVarName];
            builder.create<mlir::memref::StoreOp>( loc, unaryOp.getResult(),
                                                   memref );
            builder.create<toy::AssignOp>(
                currentAssignLoc, builder.getStringAttr( currentVarName ),
                unaryOp.getResult() );
            var_states[currentVarName] = variable_state::assigned;
            currentVarName.clear();
        }
    }

    void enterBinaryexpression(
        ToyParser::BinaryexpressionContext *ctx ) override
    {
        if ( !assignmentTargetValid )
        {
            return;
        }
        auto loc = getLocation( ctx );
        auto sz = ctx->element().size();
        auto lhs = ctx->element()[0];
        auto rhs = ctx->element()[1];
        auto op = ctx->binaryoperator()->getText();

        assert( sz == 2 );

        mlir::Value lhsValue;
        if ( lhs->INTEGERLITERAL() )
        {
            int64_t val = std::stoi( lhs->INTEGERLITERAL()->getText() );
            lhsValue =
                builder.create<mlir::arith::ConstantIntOp>( loc, val, 64 );
        }
        else if ( lhs->VARIABLENAME() )
        {
            auto varName = lhs->VARIABLENAME()->getText();
            auto varState = var_states[varName];
            if ( varState == variable_state::undeclared )
            {
                lastSemError = semantic_errors::variable_not_declared;
                llvm::errs() << std::format(
                    "{}error: Variable {} not declared in binary expr\n",
                    formatLocation( loc ), varName );
                return;
            }
            if ( varState != variable_state::assigned )
            {
                lastSemError = semantic_errors::variable_not_assigned;
                llvm::errs() << std::format(
                    "{}error: Variable {} not assigned in binary expr\n",
                    formatLocation( loc ), varName );
                return;
            }
            // Load from memref<f64>
            auto memref = var_storage[varName];
            lhsValue = builder.create<mlir::memref::LoadOp>( loc, memref );
        }
        else
        {
            lastSemError = semantic_errors::unknown_error;
            llvm::errs() << std::format(
                "{}error: Invalid binary lhs operand\n",
                formatLocation( loc ) );
            return;
        }

        mlir::Value rhsValue;
        if ( rhs->INTEGERLITERAL() )
        {
            int64_t val = std::stoi( rhs->INTEGERLITERAL()->getText() );
            rhsValue =
                builder.create<mlir::arith::ConstantIntOp>( loc, val, 64 );
        }
        else if ( rhs->VARIABLENAME() )
        {
            auto varName = rhs->VARIABLENAME()->getText();
            auto varState = var_states[varName];
            if ( varState == variable_state::undeclared )
            {
                lastSemError = semantic_errors::variable_not_declared;
                llvm::errs() << std::format(
                    "{}error: Variable {} not declared in binary expr\n",
                    formatLocation( loc ), varName );
                return;
            }
            if ( varState != variable_state::assigned )
            {
                lastSemError = semantic_errors::variable_not_assigned;
                llvm::errs() << std::format(
                    "{}error: Variable {} not assigned in binary expr\n",
                    formatLocation( loc ), varName );
                return;
            }
            // Load from memref<f64>
            auto memref = var_storage[varName];
            rhsValue = builder.create<mlir::memref::LoadOp>( loc, memref );
        }
        else
        {
            lastSemError = semantic_errors::unknown_error;
            llvm::errs() << std::format(
                "{}error: Invalid binary rhs operand\n",
                formatLocation( loc ) );
            return;
        }

        // Create BinaryOp (supports +, -, *, /)
        auto binaryOp = builder.create<toy::BinaryOp>(
            loc, builder.getF64Type(), builder.getStringAttr( op ), lhsValue,
            rhsValue );

        if ( !currentVarName.empty() )
        {
            // Store result to memref<f64>
            auto memref = var_storage[currentVarName];
            builder.create<mlir::memref::StoreOp>( loc, binaryOp.getResult(),
                                                   memref );
            builder.create<toy::AssignOp>(
                currentAssignLoc, builder.getStringAttr( currentVarName ),
                binaryOp.getResult() );
            var_states[currentVarName] = variable_state::assigned;
            currentVarName.clear();
        }
    }
};

enum class return_codes : int
{
    success,
    cannot_open_file,
    semantic_error,
    unknown_error
};

int main( int argc, char **argv )
{
    llvm::InitLLVM init( argc, argv );
    llvm::cl::ParseCommandLineOptions( argc, argv, "Calculator compiler\n" );

    std::ifstream inputStream;
    std::string filename = inputFilename;
    if ( filename != "-" )
    {
        inputStream.open( filename );
        if ( !inputStream.is_open() )
        {
            llvm::errs() << std::format( "Error: Cannot open file {}\n",
                                         filename );
            return (int)return_codes::cannot_open_file;
        }
    }
    else
    {
        filename = "input.calc";
        inputStream.basic_ios<char>::rdbuf( std::cin.rdbuf() );
    }

    try
    {
        MLIRListener listener( filename );

        antlr4::ANTLRInputStream antlrInput( inputStream );
        ToyLexer lexer( &antlrInput );
        antlr4::CommonTokenStream tokens( &lexer );
        ToyParser parser( &tokens );

        antlr4::tree::ParseTree *tree = parser.startRule();
        antlr4::tree::ParseTreeWalker::DEFAULT.walk( &listener, tree );

        listener.getModule().dump();
    }
    catch ( const semantic_exception &e )
    {
        // already printed the message.  return something non-zero
        return (int)return_codes::semantic_error;
    }
    catch ( const std::exception &e )
    {
        llvm::errs() << std::format( "FATAL ERROR: {}\n", e.what() );
        return (int)return_codes::unknown_error;
    }

    return (int)return_codes::success;
}

// vim: et ts=4 sw=4
