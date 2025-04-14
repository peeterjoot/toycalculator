#include <iostream>
#include <string>
#include <map>
#include <format>
#include "antlr4-runtime.h"
#include "calculatorLexer.h"
#include "calculatorParser.h"
#include "calculatorListener.h"
#include "calculatorBaseListener.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace antlr4;
using namespace mlir;

// Custom listener to generate MLIR
class MLIRListener : public calculatorBaseListener {
private:
    MLIRContext *context;
    OpBuilder builder;
    ModuleOp module;
    std::map<std::string, Value> variables; // Maps variable names to MLIR Values

public:
    MLIRListener(MLIRContext *ctx)
        : context(ctx), builder(ctx), module(ModuleOp::create(builder.getUnknownLoc())) {
        // Load dialects
        context->loadDialect<func::FuncDialect>();
        // Create a main function
        auto mainFunc = builder.create<func::FuncOp>(
            builder.getUnknownLoc(),
            "main",
            builder.getFunctionType({}, {}) // No args, no results
        );
        // Create an entry block
        Block *entryBlock = mainFunc.addEntryBlock();
        builder.setInsertionPointToStart(entryBlock);
    }

    // Handle declarations
    void enterDeclare(calculatorParser::DeclareContext *ctx) override {
        std::string varName = ctx->VARIABLENAME()->getText();
        std::cout << std::format("Declaration: {}\n", varName);

        // Create a toy.declare operation
        Type i32Type = builder.getI32Type();
        OperationState state(builder.getUnknownLoc(), "toy.declare");
        state.addTypes(i32Type); // Result type
        state.addAttribute("name", builder.getStringAttr(varName));
        Operation *declareOp = builder.createOperation(state);
        Value varValue = declareOp->getResult(0);
        variables[varName] = varValue;
    }

    // Handle assignments (placeholder)
    void enterAssignment(calculatorParser::AssignmentContext *ctx) override {
        std::string varName = ctx->VARIABLENAME()->getText();
        std::cout << std::format("Assignment: {} = {}\n", varName, ctx->rhs()->getText());
        // TODO: Add MLIR codegen for assignments
    }

    // Handle print statements
    void enterPrint(calculatorParser::PrintContext *ctx) override {
        std::string varName = ctx->VARIABLENAME()->getText();
        std::cout << std::format("Print: {}\n", varName);

#if 0 // This also doesn't compile, but let's figure out declare first.
        // Create a toy.print operation
        if (variables.find(varName) == variables.end()) {
            std::cerr << std::format("Error: Variable {} not declared\n", varName);
            return;
        }
        Value varValue = variables[varName];
        OperationState state(builder.getUnknownLoc(), "toy.print");
        state.addOperands(varValue);
        builder.createOperation(state);
#endif
    }

    // Get the generated module
    ModuleOp getModule() { return module; }
};

int main() {
    // Read input
    std::string input;
    std::string line;
    while (std::getline(std::cin, line)) {
        input += line + "\n";
    }

    // Parse input with ANTLR
    ANTLRInputStream inputStream(input);
    calculatorLexer lexer(&inputStream);
    CommonTokenStream tokens(&lexer);
    calculatorParser parser(&tokens);
    auto tree = parser.startRule();

    // Create MLIR context
    MLIRContext context;
    MLIRListener listener(&context);
    tree::ParseTreeWalker::DEFAULT.walk(&listener, tree);

    // Print the generated MLIR
    ModuleOp module = listener.getModule();
    module.dump();

    // Verify the module
    if (failed(verify(module))) {
        std::cerr << "MLIR verification failed" << std::endl;
        return 1;
    }

    return 0;
}
