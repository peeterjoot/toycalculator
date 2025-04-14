
#include <iostream>
#include <string>
#include <format>
#include "antlr4-runtime.h"
#include "calculatorLexer.h"
#include "calculatorParser.h"
#include "calculatorListener.h"
#include "calculatorBaseListener.h"

using namespace antlr4;

// A simple listener to print declarations and assignments
class PrintListener : public calculatorBaseListener {
public:
    void enterDeclare(calculatorParser::DeclareContext *ctx) override {
        std::cout << std::format( "Declaration: {}\n", ctx->VARIABLENAME()->getText() );
    }

    void enterPrint(calculatorParser::PrintContext *ctx) override {
        std::cout << std::format( "Print: {}\n", ctx->VARIABLENAME()->getText() );
    }

    void enterAssignment(calculatorParser::AssignmentContext *ctx) override {
        std::cout << std::format("Assignment: {} = {}\n", ctx->VARIABLENAME()->getText(), ctx->rhs()->getText() );
    }
};

int main() {
    // Read input from stdin
    std::string input;
    std::string line;
    while (std::getline(std::cin, line)) {
        input += line + "\n";
    }

    // Create ANTLR input stream
    antlr4::ANTLRInputStream inputStream(input);

    // Create lexer
    calculatorLexer lexer(&inputStream);
    CommonTokenStream tokens(&lexer);

    // Create parser
    calculatorParser parser(&tokens);

    // Parse the input starting from startRule
    auto tree = parser.startRule();

    // Walk the parse tree with the listener
    PrintListener listener;
    tree::ParseTreeWalker::DEFAULT.walk(&listener, tree);

    return 0;
}
