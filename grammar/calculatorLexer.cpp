
// Generated from calculator.g4 by ANTLR 4.13.2


#include "calculatorLexer.h"


using namespace antlr4;



using namespace antlr4;

namespace {

struct CalculatorLexerStaticData final {
  CalculatorLexerStaticData(std::vector<std::string> ruleNames,
                          std::vector<std::string> channelNames,
                          std::vector<std::string> modeNames,
                          std::vector<std::string> literalNames,
                          std::vector<std::string> symbolicNames)
      : ruleNames(std::move(ruleNames)), channelNames(std::move(channelNames)),
        modeNames(std::move(modeNames)), literalNames(std::move(literalNames)),
        symbolicNames(std::move(symbolicNames)),
        vocabulary(this->literalNames, this->symbolicNames) {}

  CalculatorLexerStaticData(const CalculatorLexerStaticData&) = delete;
  CalculatorLexerStaticData(CalculatorLexerStaticData&&) = delete;
  CalculatorLexerStaticData& operator=(const CalculatorLexerStaticData&) = delete;
  CalculatorLexerStaticData& operator=(CalculatorLexerStaticData&&) = delete;

  std::vector<antlr4::dfa::DFA> decisionToDFA;
  antlr4::atn::PredictionContextCache sharedContextCache;
  const std::vector<std::string> ruleNames;
  const std::vector<std::string> channelNames;
  const std::vector<std::string> modeNames;
  const std::vector<std::string> literalNames;
  const std::vector<std::string> symbolicNames;
  const antlr4::dfa::Vocabulary vocabulary;
  antlr4::atn::SerializedATNView serializedATN;
  std::unique_ptr<antlr4::atn::ATN> atn;
};

::antlr4::internal::OnceFlag calculatorlexerLexerOnceFlag;
#if ANTLR4_USE_THREAD_LOCAL_CACHE
static thread_local
#endif
std::unique_ptr<CalculatorLexerStaticData> calculatorlexerLexerStaticData = nullptr;

void calculatorlexerLexerInitialize() {
#if ANTLR4_USE_THREAD_LOCAL_CACHE
  if (calculatorlexerLexerStaticData != nullptr) {
    return;
  }
#else
  assert(calculatorlexerLexerStaticData == nullptr);
#endif
  auto staticData = std::make_unique<CalculatorLexerStaticData>(
    std::vector<std::string>{
      "EQUALS", "SEMICOLON", "MINUSCHAR", "TIMESCHAR", "DIVCHAR", "PLUSCHAR", 
      "DCL", "INTEGERLITERAL", "VARIABLENAME"
    },
    std::vector<std::string>{
      "DEFAULT_TOKEN_CHANNEL", "HIDDEN"
    },
    std::vector<std::string>{
      "DEFAULT_MODE"
    },
    std::vector<std::string>{
      "", "'='", "';'", "'-'", "'*'", "'/'", "'+'", "'DCL'"
    },
    std::vector<std::string>{
      "", "EQUALS", "SEMICOLON", "MINUSCHAR", "TIMESCHAR", "DIVCHAR", "PLUSCHAR", 
      "DCL", "INTEGERLITERAL", "VARIABLENAME"
    }
  );
  static const int32_t serializedATNSegment[] = {
  	4,0,9,49,6,-1,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,2,5,7,5,2,6,7,6,
  	2,7,7,7,2,8,7,8,1,0,1,0,1,1,1,1,1,2,1,2,1,3,1,3,1,4,1,4,1,5,1,5,1,6,1,
  	6,1,6,1,6,1,7,1,7,3,7,38,8,7,1,7,4,7,41,8,7,11,7,12,7,42,1,8,4,8,46,8,
  	8,11,8,12,8,47,0,0,9,1,1,3,2,5,3,7,4,9,5,11,6,13,7,15,8,17,9,1,0,2,1,
  	0,48,57,2,0,65,90,97,122,52,0,1,1,0,0,0,0,3,1,0,0,0,0,5,1,0,0,0,0,7,1,
  	0,0,0,0,9,1,0,0,0,0,11,1,0,0,0,0,13,1,0,0,0,0,15,1,0,0,0,0,17,1,0,0,0,
  	1,19,1,0,0,0,3,21,1,0,0,0,5,23,1,0,0,0,7,25,1,0,0,0,9,27,1,0,0,0,11,29,
  	1,0,0,0,13,31,1,0,0,0,15,37,1,0,0,0,17,45,1,0,0,0,19,20,5,61,0,0,20,2,
  	1,0,0,0,21,22,5,59,0,0,22,4,1,0,0,0,23,24,5,45,0,0,24,6,1,0,0,0,25,26,
  	5,42,0,0,26,8,1,0,0,0,27,28,5,47,0,0,28,10,1,0,0,0,29,30,5,43,0,0,30,
  	12,1,0,0,0,31,32,5,68,0,0,32,33,5,67,0,0,33,34,5,76,0,0,34,14,1,0,0,0,
  	35,38,3,11,5,0,36,38,3,5,2,0,37,35,1,0,0,0,37,36,1,0,0,0,37,38,1,0,0,
  	0,38,40,1,0,0,0,39,41,7,0,0,0,40,39,1,0,0,0,41,42,1,0,0,0,42,40,1,0,0,
  	0,42,43,1,0,0,0,43,16,1,0,0,0,44,46,7,1,0,0,45,44,1,0,0,0,46,47,1,0,0,
  	0,47,45,1,0,0,0,47,48,1,0,0,0,48,18,1,0,0,0,4,0,37,42,47,0
  };
  staticData->serializedATN = antlr4::atn::SerializedATNView(serializedATNSegment, sizeof(serializedATNSegment) / sizeof(serializedATNSegment[0]));

  antlr4::atn::ATNDeserializer deserializer;
  staticData->atn = deserializer.deserialize(staticData->serializedATN);

  const size_t count = staticData->atn->getNumberOfDecisions();
  staticData->decisionToDFA.reserve(count);
  for (size_t i = 0; i < count; i++) { 
    staticData->decisionToDFA.emplace_back(staticData->atn->getDecisionState(i), i);
  }
  calculatorlexerLexerStaticData = std::move(staticData);
}

}

calculatorLexer::calculatorLexer(CharStream *input) : Lexer(input) {
  calculatorLexer::initialize();
  _interpreter = new atn::LexerATNSimulator(this, *calculatorlexerLexerStaticData->atn, calculatorlexerLexerStaticData->decisionToDFA, calculatorlexerLexerStaticData->sharedContextCache);
}

calculatorLexer::~calculatorLexer() {
  delete _interpreter;
}

std::string calculatorLexer::getGrammarFileName() const {
  return "calculator.g4";
}

const std::vector<std::string>& calculatorLexer::getRuleNames() const {
  return calculatorlexerLexerStaticData->ruleNames;
}

const std::vector<std::string>& calculatorLexer::getChannelNames() const {
  return calculatorlexerLexerStaticData->channelNames;
}

const std::vector<std::string>& calculatorLexer::getModeNames() const {
  return calculatorlexerLexerStaticData->modeNames;
}

const dfa::Vocabulary& calculatorLexer::getVocabulary() const {
  return calculatorlexerLexerStaticData->vocabulary;
}

antlr4::atn::SerializedATNView calculatorLexer::getSerializedATN() const {
  return calculatorlexerLexerStaticData->serializedATN;
}

const atn::ATN& calculatorLexer::getATN() const {
  return *calculatorlexerLexerStaticData->atn;
}




void calculatorLexer::initialize() {
#if ANTLR4_USE_THREAD_LOCAL_CACHE
  calculatorlexerLexerInitialize();
#else
  ::antlr4::internal::call_once(calculatorlexerLexerOnceFlag, calculatorlexerLexerInitialize);
#endif
}
