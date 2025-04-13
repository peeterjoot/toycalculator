
// Generated from calculator.g4 by ANTLR 4.13.2

#pragma once


#include "antlr4-runtime.h"




class  calculatorParser : public antlr4::Parser {
public:
  enum {
    EQUALS = 1, SEMICOLON = 2, MINUSCHAR = 3, TIMESCHAR = 4, DIVCHAR = 5, 
    PLUSCHAR = 6, DCL = 7, INTEGERLITERAL = 8, VARIABLENAME = 9
  };

  enum {
    RuleStartrule = 0, RuleStatement = 1, RuleDeclare = 2, RuleAssignment = 3, 
    RuleRhs = 4, RuleOperator = 5, RuleElement = 6
  };

  explicit calculatorParser(antlr4::TokenStream *input);

  calculatorParser(antlr4::TokenStream *input, const antlr4::atn::ParserATNSimulatorOptions &options);

  ~calculatorParser() override;

  std::string getGrammarFileName() const override;

  const antlr4::atn::ATN& getATN() const override;

  const std::vector<std::string>& getRuleNames() const override;

  const antlr4::dfa::Vocabulary& getVocabulary() const override;

  antlr4::atn::SerializedATNView getSerializedATN() const override;


  class StartruleContext;
  class StatementContext;
  class DeclareContext;
  class AssignmentContext;
  class RhsContext;
  class OperatorContext;
  class ElementContext; 

  class  StartruleContext : public antlr4::ParserRuleContext {
  public:
    StartruleContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *EOF();
    std::vector<StatementContext *> statement();
    StatementContext* statement(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  StartruleContext* startrule();

  class  StatementContext : public antlr4::ParserRuleContext {
  public:
    StatementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<DeclareContext *> declare();
    DeclareContext* declare(size_t i);
    std::vector<AssignmentContext *> assignment();
    AssignmentContext* assignment(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  StatementContext* statement();

  class  DeclareContext : public antlr4::ParserRuleContext {
  public:
    DeclareContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *DCL();
    antlr4::tree::TerminalNode *VARIABLENAME();
    antlr4::tree::TerminalNode *SEMICOLON();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  DeclareContext* declare();

  class  AssignmentContext : public antlr4::ParserRuleContext {
  public:
    AssignmentContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *VARIABLENAME();
    antlr4::tree::TerminalNode *EQUALS();
    RhsContext *rhs();
    antlr4::tree::TerminalNode *SEMICOLON();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  AssignmentContext* assignment();

  class  RhsContext : public antlr4::ParserRuleContext {
  public:
    RhsContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<ElementContext *> element();
    ElementContext* element(size_t i);
    OperatorContext *operator_();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  RhsContext* rhs();

  class  OperatorContext : public antlr4::ParserRuleContext {
  public:
    OperatorContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *MINUSCHAR();
    antlr4::tree::TerminalNode *PLUSCHAR();
    antlr4::tree::TerminalNode *TIMESCHAR();
    antlr4::tree::TerminalNode *DIVCHAR();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  OperatorContext* operator_();

  class  ElementContext : public antlr4::ParserRuleContext {
  public:
    ElementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *INTEGERLITERAL();
    antlr4::tree::TerminalNode *VARIABLENAME();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  ElementContext* element();


  // By default the static state used to implement the parser is lazily initialized during the first
  // call to the constructor. You can call this function if you wish to initialize the static state
  // ahead of time.
  static void initialize();

private:
};

