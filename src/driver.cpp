#include "calculatorListener.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Location.h"
#include "ToyCalculatorDialect.h"
#include <antlr4-runtime.h>
#include "calculatorLexer.h"
#include "calculatorParser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include <fstream>

static llvm::cl::opt<std::string> inputFilename(
    llvm::cl::Positional, llvm::cl::desc("<input file>"),
    llvm::cl::init("-"), llvm::cl::value_desc("filename"));

class MLIRListener : public calculatorListener {
public:
    MLIRListener(mlir::OpBuilder &b, mlir::ModuleOp &m)
        : builder(b), module(m), filename("input.calc") {}

    void setFilename(const std::string &f) {
        filename = f;
    }

    mlir::Location getLocation(antlr4::ParserRuleContext *ctx) {
        size_t line = ctx->getStart()->getLine();
        size_t col = ctx->getStart()->getCharPositionInLine();
        return mlir::FileLineColLoc::get(builder.getStringAttr(filename), line, col);
    }

    void enterR(calculatorParser::RContext *ctx) override {
        auto loc = getLocation(ctx);
        builder.create<toy::ValueOp>(loc, builder.getF64Type());
    }

private:
    mlir::OpBuilder &builder;
    mlir::ModuleOp &module;
    std::string filename;
};

void processInput(std::ifstream &input, MLIRListener &listener, const std::string &filename) {
    antlr4::ANTLRInputStream antlrInput(input);
    calculatorLexer lexer(&antlrInput);
    antlr4::CommonTokenStream tokens(&lexer);
    calculatorParser parser(&tokens);

    antlr4::tree::ParseTree *tree = parser.r();
    listener.setFilename(filename);
    antlr4::tree::ParseTreeWalker::DEFAULT.walk(&listener, tree);
}

int main(int argc, char **argv) {
    llvm::InitLLVM init(argc, argv);
    llvm::cl::ParseCommandLineOptions(argc, argv, "Calculator compiler\n");

    mlir::MLIRContext context;
    context.getOrLoadDialect<toy::ToyDialect>();

    mlir::OpBuilder builder(&context);
    auto module = mlir::ModuleOp::create(builder.getUnknownLoc());

    std::ifstream inputStream;
    std::string filename = inputFilename;
    if (filename != "-") {
        inputStream.open(filename);
        if (!inputStream.is_open()) {
            llvm::errs() << "Error: Cannot open file " << filename << "\n";
            return 1;
        }
    } else {
        filename = "input.calc";
        inputStream.basic_ios<char>::rdbuf(std::cin.rdbuf());
    }

    MLIRListener listener(builder, module);
    processInput(inputStream, listener, filename);

    module.dump();

    // Verify the module
    if (failed(verify(module))) {
        std::cerr << "MLIR verification failed" << std::endl;
        return 1;
    }

    return 0;
}

// vim: et ts=4 sw=4
