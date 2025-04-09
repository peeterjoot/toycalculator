
// Generated from ToyCalculator.g4 by ANTLR 4.13.2


#include "ToyCalculatorListener.h"

#include "ToyCalculatorParser.h"


using namespace antlrcpp;

using namespace antlr4;

namespace {

struct ToyCalculatorParserStaticData final {
  ToyCalculatorParserStaticData(std::vector<std::string> ruleNames,
                        std::vector<std::string> literalNames,
                        std::vector<std::string> symbolicNames)
      : ruleNames(std::move(ruleNames)), literalNames(std::move(literalNames)),
        symbolicNames(std::move(symbolicNames)),
        vocabulary(this->literalNames, this->symbolicNames) {}

  ToyCalculatorParserStaticData(const ToyCalculatorParserStaticData&) = delete;
  ToyCalculatorParserStaticData(ToyCalculatorParserStaticData&&) = delete;
  ToyCalculatorParserStaticData& operator=(const ToyCalculatorParserStaticData&) = delete;
  ToyCalculatorParserStaticData& operator=(ToyCalculatorParserStaticData&&) = delete;

  std::vector<antlr4::dfa::DFA> decisionToDFA;
  antlr4::atn::PredictionContextCache sharedContextCache;
  const std::vector<std::string> ruleNames;
  const std::vector<std::string> literalNames;
  const std::vector<std::string> symbolicNames;
  const antlr4::dfa::Vocabulary vocabulary;
  antlr4::atn::SerializedATNView serializedATN;
  std::unique_ptr<antlr4::atn::ATN> atn;
};

::antlr4::internal::OnceFlag toycalculatorParserOnceFlag;
#if ANTLR4_USE_THREAD_LOCAL_CACHE
static thread_local
#endif
std::unique_ptr<ToyCalculatorParserStaticData> toycalculatorParserStaticData = nullptr;

void toycalculatorParserInitialize() {
#if ANTLR4_USE_THREAD_LOCAL_CACHE
  if (toycalculatorParserStaticData != nullptr) {
    return;
  }
#else
  assert(toycalculatorParserStaticData == nullptr);
#endif
  auto staticData = std::make_unique<ToyCalculatorParserStaticData>(
    std::vector<std::string>{
      "startRule", "statement", "declare", "print", "assignment", "rhs", 
      "opertype", "element"
    },
    std::vector<std::string>{
      "", "'='", "';'", "'-'", "'*'", "'/'", "'+'", "'DCL'", "'PRINT'"
    },
    std::vector<std::string>{
      "", "EQUALS", "SEMICOLON", "MINUSCHAR", "TIMESCHAR", "DIVCHAR", "PLUSCHAR", 
      "DCL", "PRINT", "INTEGERLITERAL", "VARIABLENAME", "WS"
    }
  );
  static const int32_t serializedATNSegment[] = {
  	4,1,11,53,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,2,5,7,5,2,6,7,6,2,7,
  	7,7,1,0,5,0,18,8,0,10,0,12,0,21,9,0,1,0,1,0,1,1,1,1,1,1,4,1,28,8,1,11,
  	1,12,1,29,1,2,1,2,1,2,1,2,1,3,1,3,1,3,1,3,1,4,1,4,1,4,1,4,1,4,1,5,1,5,
  	1,5,1,5,1,6,1,6,1,7,1,7,1,7,0,0,8,0,2,4,6,8,10,12,14,0,2,1,0,3,6,1,0,
  	9,10,48,0,19,1,0,0,0,2,27,1,0,0,0,4,31,1,0,0,0,6,35,1,0,0,0,8,39,1,0,
  	0,0,10,44,1,0,0,0,12,48,1,0,0,0,14,50,1,0,0,0,16,18,3,2,1,0,17,16,1,0,
  	0,0,18,21,1,0,0,0,19,17,1,0,0,0,19,20,1,0,0,0,20,22,1,0,0,0,21,19,1,0,
  	0,0,22,23,5,0,0,1,23,1,1,0,0,0,24,28,3,4,2,0,25,28,3,8,4,0,26,28,3,6,
  	3,0,27,24,1,0,0,0,27,25,1,0,0,0,27,26,1,0,0,0,28,29,1,0,0,0,29,27,1,0,
  	0,0,29,30,1,0,0,0,30,3,1,0,0,0,31,32,5,7,0,0,32,33,5,10,0,0,33,34,5,2,
  	0,0,34,5,1,0,0,0,35,36,5,8,0,0,36,37,5,10,0,0,37,38,5,2,0,0,38,7,1,0,
  	0,0,39,40,5,10,0,0,40,41,5,1,0,0,41,42,3,10,5,0,42,43,5,2,0,0,43,9,1,
  	0,0,0,44,45,3,14,7,0,45,46,3,12,6,0,46,47,3,14,7,0,47,11,1,0,0,0,48,49,
  	7,0,0,0,49,13,1,0,0,0,50,51,7,1,0,0,51,15,1,0,0,0,3,19,27,29
  };
  staticData->serializedATN = antlr4::atn::SerializedATNView(serializedATNSegment, sizeof(serializedATNSegment) / sizeof(serializedATNSegment[0]));

  antlr4::atn::ATNDeserializer deserializer;
  staticData->atn = deserializer.deserialize(staticData->serializedATN);

  const size_t count = staticData->atn->getNumberOfDecisions();
  staticData->decisionToDFA.reserve(count);
  for (size_t i = 0; i < count; i++) { 
    staticData->decisionToDFA.emplace_back(staticData->atn->getDecisionState(i), i);
  }
  toycalculatorParserStaticData = std::move(staticData);
}

}

ToyCalculatorParser::ToyCalculatorParser(TokenStream *input) : ToyCalculatorParser(input, antlr4::atn::ParserATNSimulatorOptions()) {}

ToyCalculatorParser::ToyCalculatorParser(TokenStream *input, const antlr4::atn::ParserATNSimulatorOptions &options) : Parser(input) {
  ToyCalculatorParser::initialize();
  _interpreter = new atn::ParserATNSimulator(this, *toycalculatorParserStaticData->atn, toycalculatorParserStaticData->decisionToDFA, toycalculatorParserStaticData->sharedContextCache, options);
}

ToyCalculatorParser::~ToyCalculatorParser() {
  delete _interpreter;
}

const atn::ATN& ToyCalculatorParser::getATN() const {
  return *toycalculatorParserStaticData->atn;
}

std::string ToyCalculatorParser::getGrammarFileName() const {
  return "ToyCalculator.g4";
}

const std::vector<std::string>& ToyCalculatorParser::getRuleNames() const {
  return toycalculatorParserStaticData->ruleNames;
}

const dfa::Vocabulary& ToyCalculatorParser::getVocabulary() const {
  return toycalculatorParserStaticData->vocabulary;
}

antlr4::atn::SerializedATNView ToyCalculatorParser::getSerializedATN() const {
  return toycalculatorParserStaticData->serializedATN;
}


//----------------- StartRuleContext ------------------------------------------------------------------

ToyCalculatorParser::StartRuleContext::StartRuleContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* ToyCalculatorParser::StartRuleContext::EOF() {
  return getToken(ToyCalculatorParser::EOF, 0);
}

std::vector<ToyCalculatorParser::StatementContext *> ToyCalculatorParser::StartRuleContext::statement() {
  return getRuleContexts<ToyCalculatorParser::StatementContext>();
}

ToyCalculatorParser::StatementContext* ToyCalculatorParser::StartRuleContext::statement(size_t i) {
  return getRuleContext<ToyCalculatorParser::StatementContext>(i);
}


size_t ToyCalculatorParser::StartRuleContext::getRuleIndex() const {
  return ToyCalculatorParser::RuleStartRule;
}

void ToyCalculatorParser::StartRuleContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ToyCalculatorListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterStartRule(this);
}

void ToyCalculatorParser::StartRuleContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ToyCalculatorListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitStartRule(this);
}

ToyCalculatorParser::StartRuleContext* ToyCalculatorParser::startRule() {
  StartRuleContext *_localctx = _tracker.createInstance<StartRuleContext>(_ctx, getState());
  enterRule(_localctx, 0, ToyCalculatorParser::RuleStartRule);
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
    setState(19);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & 1408) != 0)) {
      setState(16);
      statement();
      setState(21);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(22);
    match(ToyCalculatorParser::EOF);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- StatementContext ------------------------------------------------------------------

ToyCalculatorParser::StatementContext::StatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<ToyCalculatorParser::DeclareContext *> ToyCalculatorParser::StatementContext::declare() {
  return getRuleContexts<ToyCalculatorParser::DeclareContext>();
}

ToyCalculatorParser::DeclareContext* ToyCalculatorParser::StatementContext::declare(size_t i) {
  return getRuleContext<ToyCalculatorParser::DeclareContext>(i);
}

std::vector<ToyCalculatorParser::AssignmentContext *> ToyCalculatorParser::StatementContext::assignment() {
  return getRuleContexts<ToyCalculatorParser::AssignmentContext>();
}

ToyCalculatorParser::AssignmentContext* ToyCalculatorParser::StatementContext::assignment(size_t i) {
  return getRuleContext<ToyCalculatorParser::AssignmentContext>(i);
}

std::vector<ToyCalculatorParser::PrintContext *> ToyCalculatorParser::StatementContext::print() {
  return getRuleContexts<ToyCalculatorParser::PrintContext>();
}

ToyCalculatorParser::PrintContext* ToyCalculatorParser::StatementContext::print(size_t i) {
  return getRuleContext<ToyCalculatorParser::PrintContext>(i);
}


size_t ToyCalculatorParser::StatementContext::getRuleIndex() const {
  return ToyCalculatorParser::RuleStatement;
}

void ToyCalculatorParser::StatementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ToyCalculatorListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterStatement(this);
}

void ToyCalculatorParser::StatementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ToyCalculatorListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitStatement(this);
}

ToyCalculatorParser::StatementContext* ToyCalculatorParser::statement() {
  StatementContext *_localctx = _tracker.createInstance<StatementContext>(_ctx, getState());
  enterRule(_localctx, 2, ToyCalculatorParser::RuleStatement);

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
    setState(27); 
    _errHandler->sync(this);
    alt = 1;
    do {
      switch (alt) {
        case 1: {
              setState(27);
              _errHandler->sync(this);
              switch (_input->LA(1)) {
                case ToyCalculatorParser::DCL: {
                  setState(24);
                  declare();
                  break;
                }

                case ToyCalculatorParser::VARIABLENAME: {
                  setState(25);
                  assignment();
                  break;
                }

                case ToyCalculatorParser::PRINT: {
                  setState(26);
                  print();
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
      setState(29); 
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

ToyCalculatorParser::DeclareContext::DeclareContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* ToyCalculatorParser::DeclareContext::DCL() {
  return getToken(ToyCalculatorParser::DCL, 0);
}

tree::TerminalNode* ToyCalculatorParser::DeclareContext::VARIABLENAME() {
  return getToken(ToyCalculatorParser::VARIABLENAME, 0);
}

tree::TerminalNode* ToyCalculatorParser::DeclareContext::SEMICOLON() {
  return getToken(ToyCalculatorParser::SEMICOLON, 0);
}


size_t ToyCalculatorParser::DeclareContext::getRuleIndex() const {
  return ToyCalculatorParser::RuleDeclare;
}

void ToyCalculatorParser::DeclareContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ToyCalculatorListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterDeclare(this);
}

void ToyCalculatorParser::DeclareContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ToyCalculatorListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitDeclare(this);
}

ToyCalculatorParser::DeclareContext* ToyCalculatorParser::declare() {
  DeclareContext *_localctx = _tracker.createInstance<DeclareContext>(_ctx, getState());
  enterRule(_localctx, 4, ToyCalculatorParser::RuleDeclare);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(31);
    match(ToyCalculatorParser::DCL);
    setState(32);
    match(ToyCalculatorParser::VARIABLENAME);
    setState(33);
    match(ToyCalculatorParser::SEMICOLON);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PrintContext ------------------------------------------------------------------

ToyCalculatorParser::PrintContext::PrintContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* ToyCalculatorParser::PrintContext::PRINT() {
  return getToken(ToyCalculatorParser::PRINT, 0);
}

tree::TerminalNode* ToyCalculatorParser::PrintContext::VARIABLENAME() {
  return getToken(ToyCalculatorParser::VARIABLENAME, 0);
}

tree::TerminalNode* ToyCalculatorParser::PrintContext::SEMICOLON() {
  return getToken(ToyCalculatorParser::SEMICOLON, 0);
}


size_t ToyCalculatorParser::PrintContext::getRuleIndex() const {
  return ToyCalculatorParser::RulePrint;
}

void ToyCalculatorParser::PrintContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ToyCalculatorListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterPrint(this);
}

void ToyCalculatorParser::PrintContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ToyCalculatorListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitPrint(this);
}

ToyCalculatorParser::PrintContext* ToyCalculatorParser::print() {
  PrintContext *_localctx = _tracker.createInstance<PrintContext>(_ctx, getState());
  enterRule(_localctx, 6, ToyCalculatorParser::RulePrint);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(35);
    match(ToyCalculatorParser::PRINT);
    setState(36);
    match(ToyCalculatorParser::VARIABLENAME);
    setState(37);
    match(ToyCalculatorParser::SEMICOLON);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- AssignmentContext ------------------------------------------------------------------

ToyCalculatorParser::AssignmentContext::AssignmentContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* ToyCalculatorParser::AssignmentContext::VARIABLENAME() {
  return getToken(ToyCalculatorParser::VARIABLENAME, 0);
}

tree::TerminalNode* ToyCalculatorParser::AssignmentContext::EQUALS() {
  return getToken(ToyCalculatorParser::EQUALS, 0);
}

ToyCalculatorParser::RhsContext* ToyCalculatorParser::AssignmentContext::rhs() {
  return getRuleContext<ToyCalculatorParser::RhsContext>(0);
}

tree::TerminalNode* ToyCalculatorParser::AssignmentContext::SEMICOLON() {
  return getToken(ToyCalculatorParser::SEMICOLON, 0);
}


size_t ToyCalculatorParser::AssignmentContext::getRuleIndex() const {
  return ToyCalculatorParser::RuleAssignment;
}

void ToyCalculatorParser::AssignmentContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ToyCalculatorListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterAssignment(this);
}

void ToyCalculatorParser::AssignmentContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ToyCalculatorListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitAssignment(this);
}

ToyCalculatorParser::AssignmentContext* ToyCalculatorParser::assignment() {
  AssignmentContext *_localctx = _tracker.createInstance<AssignmentContext>(_ctx, getState());
  enterRule(_localctx, 8, ToyCalculatorParser::RuleAssignment);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(39);
    match(ToyCalculatorParser::VARIABLENAME);
    setState(40);
    match(ToyCalculatorParser::EQUALS);
    setState(41);
    rhs();
    setState(42);
    match(ToyCalculatorParser::SEMICOLON);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- RhsContext ------------------------------------------------------------------

ToyCalculatorParser::RhsContext::RhsContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<ToyCalculatorParser::ElementContext *> ToyCalculatorParser::RhsContext::element() {
  return getRuleContexts<ToyCalculatorParser::ElementContext>();
}

ToyCalculatorParser::ElementContext* ToyCalculatorParser::RhsContext::element(size_t i) {
  return getRuleContext<ToyCalculatorParser::ElementContext>(i);
}

ToyCalculatorParser::OpertypeContext* ToyCalculatorParser::RhsContext::opertype() {
  return getRuleContext<ToyCalculatorParser::OpertypeContext>(0);
}


size_t ToyCalculatorParser::RhsContext::getRuleIndex() const {
  return ToyCalculatorParser::RuleRhs;
}

void ToyCalculatorParser::RhsContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ToyCalculatorListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterRhs(this);
}

void ToyCalculatorParser::RhsContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ToyCalculatorListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitRhs(this);
}

ToyCalculatorParser::RhsContext* ToyCalculatorParser::rhs() {
  RhsContext *_localctx = _tracker.createInstance<RhsContext>(_ctx, getState());
  enterRule(_localctx, 10, ToyCalculatorParser::RuleRhs);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(44);
    element();
    setState(45);
    opertype();
    setState(46);
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

ToyCalculatorParser::OpertypeContext::OpertypeContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* ToyCalculatorParser::OpertypeContext::MINUSCHAR() {
  return getToken(ToyCalculatorParser::MINUSCHAR, 0);
}

tree::TerminalNode* ToyCalculatorParser::OpertypeContext::PLUSCHAR() {
  return getToken(ToyCalculatorParser::PLUSCHAR, 0);
}

tree::TerminalNode* ToyCalculatorParser::OpertypeContext::TIMESCHAR() {
  return getToken(ToyCalculatorParser::TIMESCHAR, 0);
}

tree::TerminalNode* ToyCalculatorParser::OpertypeContext::DIVCHAR() {
  return getToken(ToyCalculatorParser::DIVCHAR, 0);
}


size_t ToyCalculatorParser::OpertypeContext::getRuleIndex() const {
  return ToyCalculatorParser::RuleOpertype;
}

void ToyCalculatorParser::OpertypeContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ToyCalculatorListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterOpertype(this);
}

void ToyCalculatorParser::OpertypeContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ToyCalculatorListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitOpertype(this);
}

ToyCalculatorParser::OpertypeContext* ToyCalculatorParser::opertype() {
  OpertypeContext *_localctx = _tracker.createInstance<OpertypeContext>(_ctx, getState());
  enterRule(_localctx, 12, ToyCalculatorParser::RuleOpertype);
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
    setState(48);
    _la = _input->LA(1);
    if (!((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & 120) != 0))) {
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

ToyCalculatorParser::ElementContext::ElementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* ToyCalculatorParser::ElementContext::INTEGERLITERAL() {
  return getToken(ToyCalculatorParser::INTEGERLITERAL, 0);
}

tree::TerminalNode* ToyCalculatorParser::ElementContext::VARIABLENAME() {
  return getToken(ToyCalculatorParser::VARIABLENAME, 0);
}


size_t ToyCalculatorParser::ElementContext::getRuleIndex() const {
  return ToyCalculatorParser::RuleElement;
}

void ToyCalculatorParser::ElementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ToyCalculatorListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterElement(this);
}

void ToyCalculatorParser::ElementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<ToyCalculatorListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitElement(this);
}

ToyCalculatorParser::ElementContext* ToyCalculatorParser::element() {
  ElementContext *_localctx = _tracker.createInstance<ElementContext>(_ctx, getState());
  enterRule(_localctx, 14, ToyCalculatorParser::RuleElement);
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
    setState(50);
    _la = _input->LA(1);
    if (!(_la == ToyCalculatorParser::INTEGERLITERAL

    || _la == ToyCalculatorParser::VARIABLENAME)) {
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

void ToyCalculatorParser::initialize() {
#if ANTLR4_USE_THREAD_LOCAL_CACHE
  toycalculatorParserInitialize();
#else
  ::antlr4::internal::call_once(toycalculatorParserOnceFlag, toycalculatorParserInitialize);
#endif
}
