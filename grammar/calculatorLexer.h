
// Generated from calculator.g4 by ANTLR 4.13.2

#pragma once


#include "antlr4-runtime.h"




class  calculatorLexer : public antlr4::Lexer {
public:
  enum {
    EQUALS = 1, SEMICOLON = 2, MINUSCHAR = 3, TIMESCHAR = 4, DIVCHAR = 5, 
    PLUSCHAR = 6, DCL = 7, INTEGERLITERAL = 8, VARIABLENAME = 9
  };

  explicit calculatorLexer(antlr4::CharStream *input);

  ~calculatorLexer() override;


  std::string getGrammarFileName() const override;

  const std::vector<std::string>& getRuleNames() const override;

  const std::vector<std::string>& getChannelNames() const override;

  const std::vector<std::string>& getModeNames() const override;

  const antlr4::dfa::Vocabulary& getVocabulary() const override;

  antlr4::atn::SerializedATNView getSerializedATN() const override;

  const antlr4::atn::ATN& getATN() const override;

  // By default the static state used to implement the lexer is lazily initialized during the first
  // call to the constructor. You can call this function if you wish to initialize the static state
  // ahead of time.
  static void initialize();

private:

  // Individual action functions triggered by action() above.

  // Individual semantic predicate functions triggered by sempred() above.

};

