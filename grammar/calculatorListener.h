
// Generated from calculator.g4 by ANTLR 4.13.2

#pragma once


#include "antlr4-runtime.h"
#include "calculatorParser.h"


/**
 * This interface defines an abstract listener for a parse tree produced by calculatorParser.
 */
class  calculatorListener : public antlr4::tree::ParseTreeListener {
public:

  virtual void enterStartrule(calculatorParser::StartruleContext *ctx) = 0;
  virtual void exitStartrule(calculatorParser::StartruleContext *ctx) = 0;

  virtual void enterStatement(calculatorParser::StatementContext *ctx) = 0;
  virtual void exitStatement(calculatorParser::StatementContext *ctx) = 0;

  virtual void enterDeclare(calculatorParser::DeclareContext *ctx) = 0;
  virtual void exitDeclare(calculatorParser::DeclareContext *ctx) = 0;

  virtual void enterAssignment(calculatorParser::AssignmentContext *ctx) = 0;
  virtual void exitAssignment(calculatorParser::AssignmentContext *ctx) = 0;

  virtual void enterRhs(calculatorParser::RhsContext *ctx) = 0;
  virtual void exitRhs(calculatorParser::RhsContext *ctx) = 0;

  virtual void enterOperator(calculatorParser::OperatorContext *ctx) = 0;
  virtual void exitOperator(calculatorParser::OperatorContext *ctx) = 0;

  virtual void enterElement(calculatorParser::ElementContext *ctx) = 0;
  virtual void exitElement(calculatorParser::ElementContext *ctx) = 0;


};

