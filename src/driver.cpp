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
#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace antlr4;
using namespace mlir;

// Custom listener to generate MLIR
class MLIRListener : public calculatorBaseListener {
private:
    MLIRContext *context;
    OpBuilder builder;
    ModuleOp module;
    func::FuncOp mainFunc;
    std::map<std::string, Value> variables; // Maps variable names to MLIR Values

    // Helper to get location from parse context
    Location getLocation(ParserRuleContext *ctx) {
        size_t line = ctx->getStart()->getLine();
        size_t col = ctx->getStart()->getCharPositionInLine() + 1; // 1-based
        return builder.getFileLineColLoc(builder.getStringAttr("input.toy"), line, col);
    }

public:
    MLIRListener(MLIRContext *ctx)
        : context(ctx), builder(ctx), module(ModuleOp::create(builder.getUnknownLoc())) {
        // Load dialects
        context->loadDialect<func::FuncDialect>();
        context->loadDialect<arith::ArithDialect>();
    }

    // Set up the main function when entering startRule
    void enterStartRule(calculatorParser::StartRuleContext *ctx) override {
        // Create a main function
        mainFunc = builder.create<func::FuncOp>(
            getLocation(ctx),
            "main",
            builder.getFunctionType({}, {}) // No args, no results
        );
        // Add func to module
        module.push_back(mainFunc);
        // Create an entry block
        Block *entryBlock = mainFunc.addEntryBlock();
        builder.setInsertionPointToStart(entryBlock);
    }

    // Handle declarations
    void enterDeclare(calculatorParser::DeclareContext *ctx) override {
        std::string varName = ctx->VARIABLENAME()->getText();
        std::cout << std::format("Declaration: {}\n", varName);

        // Create an arith.constant with location
        Type i32Type = builder.getI32Type();
        auto constantOp = builder.create<arith::ConstantOp>(
            getLocation(ctx),
            i32Type,
            builder.getI32IntegerAttr(0) // Initial value 0
        );
        variables[varName] = constantOp.getResult();
    }

    // Handle assignments
    void enterAssignment(calculatorParser::AssignmentContext *ctx) override {
        std::string varName = ctx->VARIABLENAME()->getText();
        auto rhs = ctx->rhs();
        std::cout << std::format("Assignment: {} = {}\n", varName, rhs->getText());

        if (variables.find(varName) == variables.end()) {
            std::cerr << std::format("Error: Variable {} not declared\n", varName);
            return;
        }

        auto leftCtx = rhs->element(0);
        auto rightCtx = rhs->element(1);
        Value leftVal, rightVal;

        // Handle left operand
        if (leftCtx->INTEGERLITERAL()) {
            int64_t val = std::stoll(leftCtx->getText());
            leftVal = builder.create<arith::ConstantOp>(
                getLocation(leftCtx),
                builder.getI32Type(),
                builder.getI32IntegerAttr(val)
            ).getResult();
        } else {
            std::string leftName = leftCtx->VARIABLENAME()->getText();
            if (variables.find(leftName) == variables.end()) {
                std::cerr << std::format("Error: Variable {} not declared\n", leftName);
                return;
            }
            leftVal = variables[leftName];
        }

        // Handle right operand
        if (rightCtx->INTEGERLITERAL()) {
            int64_t val = std::stoll(rightCtx->getText());
            rightVal = builder.create<arith::ConstantOp>(
                getLocation(rightCtx),
                builder.getI32Type(),
                builder.getI32IntegerAttr(val)
            ).getResult();
        } else {
            std::string rightName = rightCtx->VARIABLENAME()->getText();
            if (variables.find(rightName) == variables.end()) {
                std::cerr << std::format("Error: Variable {} not declared\n", rightName);
                return;
            }
            rightVal = variables[rightName];
        }

        // Create operation based on opertype
        std::string op = rhs->opertype()->getText();
        Value resultVal;
        if (op == "+") {
            resultVal = builder.create<arith::AddIOp>(
                getLocation(ctx),
                leftVal,
                rightVal
            ).getResult();
        } else if (op == "-") {
            resultVal = builder.create<arith::SubIOp>(
                getLocation(ctx),
                leftVal,
                rightVal
            ).getResult();
        } else if (op == "*") {
            resultVal = builder.create<arith::MulIOp>(
                getLocation(ctx),
                leftVal,
                rightVal
            ).getResult();
        } else if (op == "/") {
            resultVal = builder.create<arith::DivSIOp>(
                getLocation(ctx),
                leftVal,
                rightVal
            ).getResult();
        } else {
            std::cerr << std::format("Error: Unknown operator {}\n", op);
            return;
        }

        variables[varName] = resultVal;
    }

    // Handle print statements
    void enterPrint(calculatorParser::PrintContext *ctx) override {
        std::string varName = ctx->VARIABLENAME()->getText();
        std::cout << std::format("Print: {}\n", varName);

        // Create an arith.addi with zero to simulate using the variable
        if (variables.find(varName) == variables.end()) {
            std::cerr << std::format("Error: Variable {} not declared\n", varName);
            return;
        }
        Value varValue = variables[varName];
        auto zeroOp = builder.create<arith::ConstantOp>(
            getLocation(ctx),
            builder.getI32Type(),
            builder.getI32IntegerAttr(0)
        );
        builder.create<arith::AddIOp>(
            getLocation(ctx),
            varValue,
            zeroOp.getResult()
        );
    }

    // Finalize the function with a return
    void exitStartRule(calculatorParser::StartRuleContext *ctx) override {
        builder.create<func::ReturnOp>(getLocation(ctx));
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
