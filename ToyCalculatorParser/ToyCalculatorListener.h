
// Generated from ToyCalculator.g4 by ANTLR 4.13.2

#pragma once


#include "antlr4-runtime.h"
#include "ToyCalculatorParser.h"


/**
 * This interface defines an abstract listener for a parse tree produced by ToyCalculatorParser.
 */
class  ToyCalculatorListener : public antlr4::tree::ParseTreeListener {
public:

  virtual void enterStartRule(ToyCalculatorParser::StartRuleContext *ctx) = 0;
  virtual void exitStartRule(ToyCalculatorParser::StartRuleContext *ctx) = 0;

  virtual void enterStatement(ToyCalculatorParser::StatementContext *ctx) = 0;
  virtual void exitStatement(ToyCalculatorParser::StatementContext *ctx) = 0;

  virtual void enterDeclare(ToyCalculatorParser::DeclareContext *ctx) = 0;
  virtual void exitDeclare(ToyCalculatorParser::DeclareContext *ctx) = 0;

  virtual void enterPrint(ToyCalculatorParser::PrintContext *ctx) = 0;
  virtual void exitPrint(ToyCalculatorParser::PrintContext *ctx) = 0;

  virtual void enterAssignment(ToyCalculatorParser::AssignmentContext *ctx) = 0;
  virtual void exitAssignment(ToyCalculatorParser::AssignmentContext *ctx) = 0;

  virtual void enterRhs(ToyCalculatorParser::RhsContext *ctx) = 0;
  virtual void exitRhs(ToyCalculatorParser::RhsContext *ctx) = 0;

  virtual void enterOpertype(ToyCalculatorParser::OpertypeContext *ctx) = 0;
  virtual void exitOpertype(ToyCalculatorParser::OpertypeContext *ctx) = 0;

  virtual void enterElement(ToyCalculatorParser::ElementContext *ctx) = 0;
  virtual void exitElement(ToyCalculatorParser::ElementContext *ctx) = 0;


};

