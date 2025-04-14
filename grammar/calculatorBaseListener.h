
// Generated from calculator.g4 by ANTLR 4.10

#pragma once


#include "antlr4-runtime.h"
#include "calculatorListener.h"


/**
 * This class provides an empty implementation of calculatorListener,
 * which can be extended to create a listener which only needs to handle a subset
 * of the available methods.
 */
class  calculatorBaseListener : public calculatorListener {
public:

  virtual void enterStartRule(calculatorParser::StartRuleContext * /*ctx*/) override { }
  virtual void exitStartRule(calculatorParser::StartRuleContext * /*ctx*/) override { }

  virtual void enterStatement(calculatorParser::StatementContext * /*ctx*/) override { }
  virtual void exitStatement(calculatorParser::StatementContext * /*ctx*/) override { }

  virtual void enterDeclare(calculatorParser::DeclareContext * /*ctx*/) override { }
  virtual void exitDeclare(calculatorParser::DeclareContext * /*ctx*/) override { }

  virtual void enterPrint(calculatorParser::PrintContext * /*ctx*/) override { }
  virtual void exitPrint(calculatorParser::PrintContext * /*ctx*/) override { }

  virtual void enterAssignment(calculatorParser::AssignmentContext * /*ctx*/) override { }
  virtual void exitAssignment(calculatorParser::AssignmentContext * /*ctx*/) override { }

  virtual void enterRhs(calculatorParser::RhsContext * /*ctx*/) override { }
  virtual void exitRhs(calculatorParser::RhsContext * /*ctx*/) override { }

  virtual void enterOpertype(calculatorParser::OpertypeContext * /*ctx*/) override { }
  virtual void exitOpertype(calculatorParser::OpertypeContext * /*ctx*/) override { }

  virtual void enterElement(calculatorParser::ElementContext * /*ctx*/) override { }
  virtual void exitElement(calculatorParser::ElementContext * /*ctx*/) override { }


  virtual void enterEveryRule(antlr4::ParserRuleContext * /*ctx*/) override { }
  virtual void exitEveryRule(antlr4::ParserRuleContext * /*ctx*/) override { }
  virtual void visitTerminal(antlr4::tree::TerminalNode * /*node*/) override { }
  virtual void visitErrorNode(antlr4::tree::ErrorNode * /*node*/) override { }

};

