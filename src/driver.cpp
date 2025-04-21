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
    variable_not_assigned
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
    std::unordered_map<std::string, variable_state> variables;

    mlir::Location getLocation( antlr4::ParserRuleContext *ctx )
    {
        if ( ctx )
        {
            size_t line = ctx->getStart()->getLine();
            size_t col = ctx->getStart()->getCharPositionInLine();
            return mlir::FileLineColLoc::get( builder.getStringAttr( filename ),
                                              line, col );
        }
        else
        {
            return mlir::UnknownLoc::get( &dialect.context );
        }
    }

    std::string formatLocation( mlir::Location loc )
    {
        if ( auto fileLoc = mlir::dyn_cast<mlir::FileLineColLoc>( loc ) )
        {
            return std::format( "{}:{}:{}:", fileLoc.getFilename().str(),
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
        builder.create<toy::DeclareOp>( loc, builder.getStringAttr( varName ) );
        auto varState = variables[varName];
        if ( varState == variable_state::declared )
        {
            lastSemError = semantic_errors::variable_already_declared;
            llvm::errs() << std::format(
                "{}error: Variable {} already declared in DCL\n",
                formatLocation( loc ), varName );
        }
        else if ( varState == variable_state::undeclared )
        {
            variables[varName] = variable_state::declared;
        }
    }

    void enterPrint( ToyParser::PrintContext *ctx ) override
    {
        auto loc = getLocation( ctx );
        auto varName = ctx->VARIABLENAME()->getText();
        auto varState = variables[varName];
        if ( varState != variable_state::assigned )
        {
            lastSemError = semantic_errors::variable_not_assigned;
            llvm::errs() << std::format(
                "{}error: Variable {} not assigned in PRINT\n",
                formatLocation( loc ), varName );
        }
        builder.create<toy::PrintOp>( loc, builder.getStringAttr( varName ) );
    }

    void enterAssignment( ToyParser::AssignmentContext *ctx ) override
    {
        auto loc = getLocation( ctx );
        currentVarName = ctx->VARIABLENAME()->getText();
        auto varState = variables[currentVarName];
        if ( varState == variable_state::declared )
        {
            variables[currentVarName] = variable_state::assigned;
        }
        else if ( varState != variable_state::declared )
        {
            lastSemError = semantic_errors::variable_not_declared;
            llvm::errs() << std::format(
                "{}error: Variable {} not declared in assignment\n",
                formatLocation( loc ), currentVarName );
        }
        currentAssignLoc = loc;
    }

    void enterUnaryexpression( ToyParser::UnaryexpressionContext *ctx ) override
    {
        auto lhs = ctx->element();
        auto op = ctx->unaryoperator()->getText();

        mlir::Value lhsValue;
        if ( lhs->INTEGERLITERAL() )
        {
            int64_t val = std::stoi( lhs->INTEGERLITERAL()->getText() );
            lhsValue = builder.create<mlir::arith::ConstantIntOp>(
                getLocation( lhs ), val, 64 );
        }
        else
        {
            llvm::errs() << std::format(
                "Warning: Variable {} not supported at line {}\n",
                lhs->VARIABLENAME()->getText(), ctx->getStart()->getLine() );
            return;
        }

        // FIXME: what about op == ""?
        auto unaryOp = builder.create<toy::UnaryOp>(
            getLocation( ctx ), builder.getF64Type(),
            builder.getStringAttr( op ), lhsValue );

        if ( !currentVarName.empty() )
        {
            builder.create<toy::AssignOp>(
                currentAssignLoc, builder.getStringAttr( currentVarName ),
                unaryOp.getResult() );
            currentVarName.clear();
        }
    }

    void enterBinaryexpression(
        ToyParser::BinaryexpressionContext *ctx ) override
    {
        auto sz = ctx->element().size();
        auto lhs = ctx->element()[0];
        auto rhs = ctx->element()[1];
        auto op = ctx->binaryoperator()->getText();

        assert( sz == 2 );

        mlir::Value lhsValue;
        if ( lhs->INTEGERLITERAL() )
        {
            int64_t val = std::stoi( lhs->INTEGERLITERAL()->getText() );
            lhsValue = builder.create<mlir::arith::ConstantIntOp>(
                getLocation( lhs ), val, 64 );
        }
        else
        {
            llvm::errs() << std::format(
                "Warning: Variable {} not supported at line {}\n",
                lhs->VARIABLENAME()->getText(), ctx->getStart()->getLine() );
            return;
        }

        mlir::Value rhsValue;
        if ( rhs->INTEGERLITERAL() )
        {
            int64_t val = std::stoi( rhs->INTEGERLITERAL()->getText() );
            rhsValue = builder.create<mlir::arith::ConstantIntOp>(
                getLocation( rhs ), val, 64 );
        }
        else
        {
            llvm::errs() << std::format(
                "Warning: Variable {} not supported at line {}\n",
                rhs->VARIABLENAME()->getText(), ctx->getStart()->getLine() );
            return;
        }

        auto binaryOp = builder.create<toy::BinaryOp>(
            getLocation( ctx ), builder.getF64Type(),
            builder.getStringAttr( op ), lhsValue, rhsValue );
        if ( !currentVarName.empty() )
        {
            builder.create<toy::AssignOp>(
                currentAssignLoc, builder.getStringAttr( currentVarName ),
                binaryOp.getResult() );
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
