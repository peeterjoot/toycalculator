
// Generated from ToyCalculator.g4 by ANTLR 4.13.2

#pragma once


#include "antlr4-runtime.h"
#include "ToyCalculatorListener.h"


/**
 * This class provides an empty implementation of ToyCalculatorListener,
 * which can be extended to create a listener which only needs to handle a subset
 * of the available methods.
 */
class  ToyCalculatorBaseListener : public ToyCalculatorListener {
public:

  virtual void enterStartRule(ToyCalculatorParser::StartRuleContext * /*ctx*/) override { }
  virtual void exitStartRule(ToyCalculatorParser::StartRuleContext * /*ctx*/) override { }

  virtual void enterStatement(ToyCalculatorParser::StatementContext * /*ctx*/) override { }
  virtual void exitStatement(ToyCalculatorParser::StatementContext * /*ctx*/) override { }

  virtual void enterDeclare(ToyCalculatorParser::DeclareContext * /*ctx*/) override { }
  virtual void exitDeclare(ToyCalculatorParser::DeclareContext * /*ctx*/) override { }

  virtual void enterPrint(ToyCalculatorParser::PrintContext * /*ctx*/) override { }
  virtual void exitPrint(ToyCalculatorParser::PrintContext * /*ctx*/) override { }

  virtual void enterAssignment(ToyCalculatorParser::AssignmentContext * /*ctx*/) override { }
  virtual void exitAssignment(ToyCalculatorParser::AssignmentContext * /*ctx*/) override { }

  virtual void enterRhs(ToyCalculatorParser::RhsContext * /*ctx*/) override { }
  virtual void exitRhs(ToyCalculatorParser::RhsContext * /*ctx*/) override { }

  virtual void enterOpertype(ToyCalculatorParser::OpertypeContext * /*ctx*/) override { }
  virtual void exitOpertype(ToyCalculatorParser::OpertypeContext * /*ctx*/) override { }

  virtual void enterElement(ToyCalculatorParser::ElementContext * /*ctx*/) override { }
  virtual void exitElement(ToyCalculatorParser::ElementContext * /*ctx*/) override { }


  virtual void enterEveryRule(antlr4::ParserRuleContext * /*ctx*/) override { }
  virtual void exitEveryRule(antlr4::ParserRuleContext * /*ctx*/) override { }
  virtual void visitTerminal(antlr4::tree::TerminalNode * /*node*/) override { }
  virtual void visitErrorNode(antlr4::tree::ErrorNode * /*node*/) override { }

};

