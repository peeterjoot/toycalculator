
// Generated from calculator.g4 by ANTLR 4.10


#include "calculatorListener.h"

#include "calculatorParser.h"


using namespace antlrcpp;

using namespace antlr4;

namespace {

struct CalculatorParserStaticData final {
  CalculatorParserStaticData(std::vector<std::string> ruleNames,
                        std::vector<std::string> literalNames,
                        std::vector<std::string> symbolicNames)
      : ruleNames(std::move(ruleNames)), literalNames(std::move(literalNames)),
        symbolicNames(std::move(symbolicNames)),
        vocabulary(this->literalNames, this->symbolicNames) {}

  CalculatorParserStaticData(const CalculatorParserStaticData&) = delete;
  CalculatorParserStaticData(CalculatorParserStaticData&&) = delete;
  CalculatorParserStaticData& operator=(const CalculatorParserStaticData&) = delete;
  CalculatorParserStaticData& operator=(CalculatorParserStaticData&&) = delete;

  std::vector<antlr4::dfa::DFA> decisionToDFA;
  antlr4::atn::PredictionContextCache sharedContextCache;
  const std::vector<std::string> ruleNames;
  const std::vector<std::string> literalNames;
  const std::vector<std::string> symbolicNames;
  const antlr4::dfa::Vocabulary vocabulary;
  antlr4::atn::SerializedATNView serializedATN;
  std::unique_ptr<antlr4::atn::ATN> atn;
};

std::once_flag calculatorParserOnceFlag;
CalculatorParserStaticData *calculatorParserStaticData = nullptr;

void calculatorParserInitialize() {
  assert(calculatorParserStaticData == nullptr);
  auto staticData = std::make_unique<CalculatorParserStaticData>(
    std::vector<std::string>{
      "startRule", "statement", "declare", "assignment", "rhs", "opertype", 
      "element"
    },
    std::vector<std::string>{
      "", "'='", "';'", "'-'", "'*'", "'/'", "'+'", "'DCL'"
    },
    std::vector<std::string>{
      "", "EQUALS", "SEMICOLON", "MINUSCHAR", "TIMESCHAR", "DIVCHAR", "PLUSCHAR", 
      "DCL", "INTEGERLITERAL", "VARIABLENAME", "WS"
    }
  );
  static const int32_t serializedATNSegment[] = {
  	4,1,10,46,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,2,5,7,5,2,6,7,6,1,0,
  	5,0,16,8,0,10,0,12,0,19,9,0,1,0,1,0,1,1,1,1,4,1,25,8,1,11,1,12,1,26,1,
  	2,1,2,1,2,1,2,1,3,1,3,1,3,1,3,1,3,1,4,1,4,1,4,1,4,1,5,1,5,1,6,1,6,1,6,
  	0,0,7,0,2,4,6,8,10,12,0,2,1,0,3,6,1,0,8,9,41,0,17,1,0,0,0,2,24,1,0,0,
  	0,4,28,1,0,0,0,6,32,1,0,0,0,8,37,1,0,0,0,10,41,1,0,0,0,12,43,1,0,0,0,
  	14,16,3,2,1,0,15,14,1,0,0,0,16,19,1,0,0,0,17,15,1,0,0,0,17,18,1,0,0,0,
  	18,20,1,0,0,0,19,17,1,0,0,0,20,21,5,0,0,1,21,1,1,0,0,0,22,25,3,4,2,0,
  	23,25,3,6,3,0,24,22,1,0,0,0,24,23,1,0,0,0,25,26,1,0,0,0,26,24,1,0,0,0,
  	26,27,1,0,0,0,27,3,1,0,0,0,28,29,5,7,0,0,29,30,5,9,0,0,30,31,5,2,0,0,
  	31,5,1,0,0,0,32,33,5,9,0,0,33,34,5,1,0,0,34,35,3,8,4,0,35,36,5,2,0,0,
  	36,7,1,0,0,0,37,38,3,12,6,0,38,39,3,10,5,0,39,40,3,12,6,0,40,9,1,0,0,
  	0,41,42,7,0,0,0,42,11,1,0,0,0,43,44,7,1,0,0,44,13,1,0,0,0,3,17,24,26
  };
  staticData->serializedATN = antlr4::atn::SerializedATNView(serializedATNSegment, sizeof(serializedATNSegment) / sizeof(serializedATNSegment[0]));

  antlr4::atn::ATNDeserializer deserializer;
  staticData->atn = deserializer.deserialize(staticData->serializedATN);

  const size_t count = staticData->atn->getNumberOfDecisions();
  staticData->decisionToDFA.reserve(count);
  for (size_t i = 0; i < count; i++) { 
    staticData->decisionToDFA.emplace_back(staticData->atn->getDecisionState(i), i);
  }
  calculatorParserStaticData = staticData.release();
}

}

calculatorParser::calculatorParser(TokenStream *input) : calculatorParser(input, antlr4::atn::ParserATNSimulatorOptions()) {}

calculatorParser::calculatorParser(TokenStream *input, const antlr4::atn::ParserATNSimulatorOptions &options) : Parser(input) {
  calculatorParser::initialize();
  _interpreter = new atn::ParserATNSimulator(this, *calculatorParserStaticData->atn, calculatorParserStaticData->decisionToDFA, calculatorParserStaticData->sharedContextCache, options);
}

calculatorParser::~calculatorParser() {
  delete _interpreter;
}

const atn::ATN& calculatorParser::getATN() const {
  return *calculatorParserStaticData->atn;
}

std::string calculatorParser::getGrammarFileName() const {
  return "calculator.g4";
}

const std::vector<std::string>& calculatorParser::getRuleNames() const {
  return calculatorParserStaticData->ruleNames;
}

const dfa::Vocabulary& calculatorParser::getVocabulary() const {
  return calculatorParserStaticData->vocabulary;
}

antlr4::atn::SerializedATNView calculatorParser::getSerializedATN() const {
  return calculatorParserStaticData->serializedATN;
}


//----------------- StartRuleContext ------------------------------------------------------------------

calculatorParser::StartRuleContext::StartRuleContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* calculatorParser::StartRuleContext::EOF() {
  return getToken(calculatorParser::EOF, 0);
}

std::vector<calculatorParser::StatementContext *> calculatorParser::StartRuleContext::statement() {
  return getRuleContexts<calculatorParser::StatementContext>();
}

calculatorParser::StatementContext* calculatorParser::StartRuleContext::statement(size_t i) {
  return getRuleContext<calculatorParser::StatementContext>(i);
}


size_t calculatorParser::StartRuleContext::getRuleIndex() const {
  return calculatorParser::RuleStartRule;
}

void calculatorParser::StartRuleContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<calculatorListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterStartRule(this);
}

void calculatorParser::StartRuleContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<calculatorListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitStartRule(this);
}

calculatorParser::StartRuleContext* calculatorParser::startRule() {
  StartRuleContext *_localctx = _tracker.createInstance<StartRuleContext>(_ctx, getState());
  enterRule(_localctx, 0, calculatorParser::RuleStartRule);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(17);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == calculatorParser::DCL

    || _la == calculatorParser::VARIABLENAME) {
      setState(14);
      statement();
      setState(19);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(20);
    match(calculatorParser::EOF);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- StatementContext ------------------------------------------------------------------

calculatorParser::StatementContext::StatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<calculatorParser::DeclareContext *> calculatorParser::StatementContext::declare() {
  return getRuleContexts<calculatorParser::DeclareContext>();
}

calculatorParser::DeclareContext* calculatorParser::StatementContext::declare(size_t i) {
  return getRuleContext<calculatorParser::DeclareContext>(i);
}

std::vector<calculatorParser::AssignmentContext *> calculatorParser::StatementContext::assignment() {
  return getRuleContexts<calculatorParser::AssignmentContext>();
}

calculatorParser::AssignmentContext* calculatorParser::StatementContext::assignment(size_t i) {
  return getRuleContext<calculatorParser::AssignmentContext>(i);
}


size_t calculatorParser::StatementContext::getRuleIndex() const {
  return calculatorParser::RuleStatement;
}

void calculatorParser::StatementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<calculatorListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterStatement(this);
}

void calculatorParser::StatementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<calculatorListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitStatement(this);
}

calculatorParser::StatementContext* calculatorParser::statement() {
  StatementContext *_localctx = _tracker.createInstance<StatementContext>(_ctx, getState());
  enterRule(_localctx, 2, calculatorParser::RuleStatement);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(24); 
    _errHandler->sync(this);
    alt = 1;
    do {
      switch (alt) {
        case 1: {
              setState(24);
              _errHandler->sync(this);
              switch (_input->LA(1)) {
                case calculatorParser::DCL: {
                  setState(22);
                  declare();
                  break;
                }

                case calculatorParser::VARIABLENAME: {
                  setState(23);
                  assignment();
                  break;
                }

              default:
                throw NoViableAltException(this);
              }
              break;
            }

      default:
        throw NoViableAltException(this);
      }
      setState(26); 
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 2, _ctx);
    } while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- DeclareContext ------------------------------------------------------------------

calculatorParser::DeclareContext::DeclareContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* calculatorParser::DeclareContext::DCL() {
  return getToken(calculatorParser::DCL, 0);
}

tree::TerminalNode* calculatorParser::DeclareContext::VARIABLENAME() {
  return getToken(calculatorParser::VARIABLENAME, 0);
}

tree::TerminalNode* calculatorParser::DeclareContext::SEMICOLON() {
  return getToken(calculatorParser::SEMICOLON, 0);
}


size_t calculatorParser::DeclareContext::getRuleIndex() const {
  return calculatorParser::RuleDeclare;
}

void calculatorParser::DeclareContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<calculatorListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterDeclare(this);
}

void calculatorParser::DeclareContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<calculatorListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitDeclare(this);
}

calculatorParser::DeclareContext* calculatorParser::declare() {
  DeclareContext *_localctx = _tracker.createInstance<DeclareContext>(_ctx, getState());
  enterRule(_localctx, 4, calculatorParser::RuleDeclare);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(28);
    match(calculatorParser::DCL);
    setState(29);
    match(calculatorParser::VARIABLENAME);
    setState(30);
    match(calculatorParser::SEMICOLON);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- AssignmentContext ------------------------------------------------------------------

calculatorParser::AssignmentContext::AssignmentContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* calculatorParser::AssignmentContext::VARIABLENAME() {
  return getToken(calculatorParser::VARIABLENAME, 0);
}

tree::TerminalNode* calculatorParser::AssignmentContext::EQUALS() {
  return getToken(calculatorParser::EQUALS, 0);
}

calculatorParser::RhsContext* calculatorParser::AssignmentContext::rhs() {
  return getRuleContext<calculatorParser::RhsContext>(0);
}

tree::TerminalNode* calculatorParser::AssignmentContext::SEMICOLON() {
  return getToken(calculatorParser::SEMICOLON, 0);
}


size_t calculatorParser::AssignmentContext::getRuleIndex() const {
  return calculatorParser::RuleAssignment;
}

void calculatorParser::AssignmentContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<calculatorListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterAssignment(this);
}

void calculatorParser::AssignmentContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<calculatorListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitAssignment(this);
}

calculatorParser::AssignmentContext* calculatorParser::assignment() {
  AssignmentContext *_localctx = _tracker.createInstance<AssignmentContext>(_ctx, getState());
  enterRule(_localctx, 6, calculatorParser::RuleAssignment);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(32);
    match(calculatorParser::VARIABLENAME);
    setState(33);
    match(calculatorParser::EQUALS);
    setState(34);
    rhs();
    setState(35);
    match(calculatorParser::SEMICOLON);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- RhsContext ------------------------------------------------------------------

calculatorParser::RhsContext::RhsContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<calculatorParser::ElementContext *> calculatorParser::RhsContext::element() {
  return getRuleContexts<calculatorParser::ElementContext>();
}

calculatorParser::ElementContext* calculatorParser::RhsContext::element(size_t i) {
  return getRuleContext<calculatorParser::ElementContext>(i);
}

calculatorParser::OpertypeContext* calculatorParser::RhsContext::opertype() {
  return getRuleContext<calculatorParser::OpertypeContext>(0);
}


size_t calculatorParser::RhsContext::getRuleIndex() const {
  return calculatorParser::RuleRhs;
}

void calculatorParser::RhsContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<calculatorListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterRhs(this);
}

void calculatorParser::RhsContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<calculatorListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitRhs(this);
}

calculatorParser::RhsContext* calculatorParser::rhs() {
  RhsContext *_localctx = _tracker.createInstance<RhsContext>(_ctx, getState());
  enterRule(_localctx, 8, calculatorParser::RuleRhs);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(37);
    element();
    setState(38);
    opertype();
    setState(39);
    element();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- OpertypeContext ------------------------------------------------------------------

calculatorParser::OpertypeContext::OpertypeContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* calculatorParser::OpertypeContext::MINUSCHAR() {
  return getToken(calculatorParser::MINUSCHAR, 0);
}

tree::TerminalNode* calculatorParser::OpertypeContext::PLUSCHAR() {
  return getToken(calculatorParser::PLUSCHAR, 0);
}

tree::TerminalNode* calculatorParser::OpertypeContext::TIMESCHAR() {
  return getToken(calculatorParser::TIMESCHAR, 0);
}

tree::TerminalNode* calculatorParser::OpertypeContext::DIVCHAR() {
  return getToken(calculatorParser::DIVCHAR, 0);
}


size_t calculatorParser::OpertypeContext::getRuleIndex() const {
  return calculatorParser::RuleOpertype;
}

void calculatorParser::OpertypeContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<calculatorListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterOpertype(this);
}

void calculatorParser::OpertypeContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<calculatorListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitOpertype(this);
}

calculatorParser::OpertypeContext* calculatorParser::opertype() {
  OpertypeContext *_localctx = _tracker.createInstance<OpertypeContext>(_ctx, getState());
  enterRule(_localctx, 10, calculatorParser::RuleOpertype);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(41);
    _la = _input->LA(1);
    if (!((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << calculatorParser::MINUSCHAR)
      | (1ULL << calculatorParser::TIMESCHAR)
      | (1ULL << calculatorParser::DIVCHAR)
      | (1ULL << calculatorParser::PLUSCHAR))) != 0))) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ElementContext ------------------------------------------------------------------

calculatorParser::ElementContext::ElementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* calculatorParser::ElementContext::INTEGERLITERAL() {
  return getToken(calculatorParser::INTEGERLITERAL, 0);
}

tree::TerminalNode* calculatorParser::ElementContext::VARIABLENAME() {
  return getToken(calculatorParser::VARIABLENAME, 0);
}


size_t calculatorParser::ElementContext::getRuleIndex() const {
  return calculatorParser::RuleElement;
}

void calculatorParser::ElementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<calculatorListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterElement(this);
}

void calculatorParser::ElementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<calculatorListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitElement(this);
}

calculatorParser::ElementContext* calculatorParser::element() {
  ElementContext *_localctx = _tracker.createInstance<ElementContext>(_ctx, getState());
  enterRule(_localctx, 12, calculatorParser::RuleElement);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(43);
    _la = _input->LA(1);
    if (!(_la == calculatorParser::INTEGERLITERAL

    || _la == calculatorParser::VARIABLENAME)) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

void calculatorParser::initialize() {
  std::call_once(calculatorParserOnceFlag, calculatorParserInitialize);
}
