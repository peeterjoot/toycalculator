#include <antlr4-runtime.h>

#include <format>
#include <fstream>
#include <assert.h>

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

class MLIRListener : public ToyBaseListener
{
   public:
    MLIRListener( mlir::OpBuilder &b, mlir::ModuleOp &m,
                  const std::string &_filename )
        : builder( b ),
          module( m ),
          filename( _filename ),
          currentAssignLoc( b.getUnknownLoc() )
    {
    }

    mlir::Location getLocation( antlr4::ParserRuleContext *ctx )
    {
        size_t line = ctx->getStart()->getLine();
        size_t col = ctx->getStart()->getCharPositionInLine();
        return mlir::FileLineColLoc::get( builder.getStringAttr( filename ),
                                          line, col );
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
        builder.setInsertionPointToEnd( module.getBody() );
    }

    void enterDeclare( ToyParser::DeclareContext *ctx ) override
    {
        auto loc = getLocation( ctx );
        auto varName = ctx->VARIABLENAME()->getText();
        builder.create<toy::DeclareOp>( loc, builder.getStringAttr( varName ) );
    }

    void enterPrint( ToyParser::PrintContext *ctx ) override
    {
        auto loc = getLocation( ctx );
        auto varName = ctx->VARIABLENAME()->getText();
        builder.create<toy::PrintOp>( loc, builder.getStringAttr( varName ) );
    }

    void enterAssignment( ToyParser::AssignmentContext *ctx ) override
    {
        auto loc = getLocation( ctx );
        currentVarName = ctx->VARIABLENAME()->getText();
        currentAssignLoc = loc;
    }

    void enterRhs( ToyParser::RhsContext *ctx ) override
    {
        auto sz = ctx->element().size();
        auto lhs = ctx->element()[0];

        assert( sz == 1 || sz == 2 );

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

        if ( sz == 1 ) {
            auto unaryOp = builder.create<toy::UnaryOp>(
                getLocation( ctx ), builder.getF64Type(),
                builder.getStringAttr( "+" ), lhsValue ); // fake a unary + for now.  generalize this appropriately later.
            if ( !currentVarName.empty() )
            {
                builder.create<toy::AssignOp>(
                    currentAssignLoc, builder.getStringAttr( currentVarName ),
                    unaryOp.getResult() );
                currentVarName.clear();
            }
        } else {
            auto rhs = ctx->element()[1];
            auto op = ctx->opertype()->getText();

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
    }

   private:
    mlir::OpBuilder &builder;
    mlir::ModuleOp &module;
    std::string filename;
    toy::ProgramOp programOp;
    std::string currentVarName;
    mlir::Location currentAssignLoc;
};

void processInput( std::ifstream &input, MLIRListener &listener )
{
    antlr4::ANTLRInputStream antlrInput( input );
    ToyLexer lexer( &antlrInput );
    antlr4::CommonTokenStream tokens( &lexer );
    ToyParser parser( &tokens );

    antlr4::tree::ParseTree *tree = parser.startRule();
    antlr4::tree::ParseTreeWalker::DEFAULT.walk( &listener, tree );
}

int main( int argc, char **argv )
{
    llvm::InitLLVM init( argc, argv );
    llvm::cl::ParseCommandLineOptions( argc, argv, "Calculator compiler\n" );

#if 0
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
        llvm::MemoryBuffer::getFileOrSTDIN( inputFilename );

    if ( std::error_code ec = fileOrErr.getError() )
    {
        llvm::errs() << "Could not open input file: " << ec.message() << "\n";
        return 1;
    }
#endif

    mlir::MLIRContext context;
    context.getOrLoadDialect<toy::ToyDialect>();
    context.getOrLoadDialect<mlir::arith::ArithDialect>();

    mlir::OpBuilder builder( &context );
    auto module = mlir::ModuleOp::create( builder.getUnknownLoc() );

    std::ifstream inputStream;
    std::string filename = inputFilename;
    if ( filename != "-" )
    {
        inputStream.open( filename );
        if ( !inputStream.is_open() )
        {
            llvm::errs() << std::format( "Error: Cannot open file {}\n",
                                         filename );
            return 1;
        }
    }
    else
    {
        filename = "input.calc";
        inputStream.basic_ios<char>::rdbuf( std::cin.rdbuf() );
    }

    MLIRListener listener( builder, module, filename );
    processInput( inputStream, listener );

    module.dump();

    return 0;
}

// vim: et ts=4 sw=4
