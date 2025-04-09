
// Generated from ToyCalculator.g4 by ANTLR 4.13.2


#include "ToyCalculatorLexer.h"


using namespace antlr4;



using namespace antlr4;

namespace {

struct ToyCalculatorLexerStaticData final {
  ToyCalculatorLexerStaticData(std::vector<std::string> ruleNames,
                          std::vector<std::string> channelNames,
                          std::vector<std::string> modeNames,
                          std::vector<std::string> literalNames,
                          std::vector<std::string> symbolicNames)
      : ruleNames(std::move(ruleNames)), channelNames(std::move(channelNames)),
        modeNames(std::move(modeNames)), literalNames(std::move(literalNames)),
        symbolicNames(std::move(symbolicNames)),
        vocabulary(this->literalNames, this->symbolicNames) {}

  ToyCalculatorLexerStaticData(const ToyCalculatorLexerStaticData&) = delete;
  ToyCalculatorLexerStaticData(ToyCalculatorLexerStaticData&&) = delete;
  ToyCalculatorLexerStaticData& operator=(const ToyCalculatorLexerStaticData&) = delete;
  ToyCalculatorLexerStaticData& operator=(ToyCalculatorLexerStaticData&&) = delete;

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

::antlr4::internal::OnceFlag toycalculatorlexerLexerOnceFlag;
#if ANTLR4_USE_THREAD_LOCAL_CACHE
static thread_local
#endif
std::unique_ptr<ToyCalculatorLexerStaticData> toycalculatorlexerLexerStaticData = nullptr;

void toycalculatorlexerLexerInitialize() {
#if ANTLR4_USE_THREAD_LOCAL_CACHE
  if (toycalculatorlexerLexerStaticData != nullptr) {
    return;
  }
#else
  assert(toycalculatorlexerLexerStaticData == nullptr);
#endif
  auto staticData = std::make_unique<ToyCalculatorLexerStaticData>(
    std::vector<std::string>{
      "EQUALS", "SEMICOLON", "MINUSCHAR", "TIMESCHAR", "DIVCHAR", "PLUSCHAR", 
      "DCL", "PRINT", "INTEGERLITERAL", "VARIABLENAME", "WS"
    },
    std::vector<std::string>{
      "DEFAULT_TOKEN_CHANNEL", "HIDDEN"
    },
    std::vector<std::string>{
      "DEFAULT_MODE"
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
  	4,0,11,66,6,-1,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,2,5,7,5,2,6,7,
  	6,2,7,7,7,2,8,7,8,2,9,7,9,2,10,7,10,1,0,1,0,1,1,1,1,1,2,1,2,1,3,1,3,1,
  	4,1,4,1,5,1,5,1,6,1,6,1,6,1,6,1,7,1,7,1,7,1,7,1,7,1,7,1,8,1,8,3,8,48,
  	8,8,1,8,4,8,51,8,8,11,8,12,8,52,1,9,4,9,56,8,9,11,9,12,9,57,1,10,4,10,
  	61,8,10,11,10,12,10,62,1,10,1,10,0,0,11,1,1,3,2,5,3,7,4,9,5,11,6,13,7,
  	15,8,17,9,19,10,21,11,1,0,3,1,0,48,57,2,0,65,90,97,122,3,0,9,10,13,13,
  	32,32,70,0,1,1,0,0,0,0,3,1,0,0,0,0,5,1,0,0,0,0,7,1,0,0,0,0,9,1,0,0,0,
  	0,11,1,0,0,0,0,13,1,0,0,0,0,15,1,0,0,0,0,17,1,0,0,0,0,19,1,0,0,0,0,21,
  	1,0,0,0,1,23,1,0,0,0,3,25,1,0,0,0,5,27,1,0,0,0,7,29,1,0,0,0,9,31,1,0,
  	0,0,11,33,1,0,0,0,13,35,1,0,0,0,15,39,1,0,0,0,17,47,1,0,0,0,19,55,1,0,
  	0,0,21,60,1,0,0,0,23,24,5,61,0,0,24,2,1,0,0,0,25,26,5,59,0,0,26,4,1,0,
  	0,0,27,28,5,45,0,0,28,6,1,0,0,0,29,30,5,42,0,0,30,8,1,0,0,0,31,32,5,47,
  	0,0,32,10,1,0,0,0,33,34,5,43,0,0,34,12,1,0,0,0,35,36,5,68,0,0,36,37,5,
  	67,0,0,37,38,5,76,0,0,38,14,1,0,0,0,39,40,5,80,0,0,40,41,5,82,0,0,41,
  	42,5,73,0,0,42,43,5,78,0,0,43,44,5,84,0,0,44,16,1,0,0,0,45,48,3,11,5,
  	0,46,48,3,5,2,0,47,45,1,0,0,0,47,46,1,0,0,0,47,48,1,0,0,0,48,50,1,0,0,
  	0,49,51,7,0,0,0,50,49,1,0,0,0,51,52,1,0,0,0,52,50,1,0,0,0,52,53,1,0,0,
  	0,53,18,1,0,0,0,54,56,7,1,0,0,55,54,1,0,0,0,56,57,1,0,0,0,57,55,1,0,0,
  	0,57,58,1,0,0,0,58,20,1,0,0,0,59,61,7,2,0,0,60,59,1,0,0,0,61,62,1,0,0,
  	0,62,60,1,0,0,0,62,63,1,0,0,0,63,64,1,0,0,0,64,65,6,10,0,0,65,22,1,0,
  	0,0,5,0,47,52,57,62,1,6,0,0
  };
  staticData->serializedATN = antlr4::atn::SerializedATNView(serializedATNSegment, sizeof(serializedATNSegment) / sizeof(serializedATNSegment[0]));

  antlr4::atn::ATNDeserializer deserializer;
  staticData->atn = deserializer.deserialize(staticData->serializedATN);

  const size_t count = staticData->atn->getNumberOfDecisions();
  staticData->decisionToDFA.reserve(count);
  for (size_t i = 0; i < count; i++) { 
    staticData->decisionToDFA.emplace_back(staticData->atn->getDecisionState(i), i);
  }
  toycalculatorlexerLexerStaticData = std::move(staticData);
}

}

ToyCalculatorLexer::ToyCalculatorLexer(CharStream *input) : Lexer(input) {
  ToyCalculatorLexer::initialize();
  _interpreter = new atn::LexerATNSimulator(this, *toycalculatorlexerLexerStaticData->atn, toycalculatorlexerLexerStaticData->decisionToDFA, toycalculatorlexerLexerStaticData->sharedContextCache);
}

ToyCalculatorLexer::~ToyCalculatorLexer() {
  delete _interpreter;
}

std::string ToyCalculatorLexer::getGrammarFileName() const {
  return "ToyCalculator.g4";
}

const std::vector<std::string>& ToyCalculatorLexer::getRuleNames() const {
  return toycalculatorlexerLexerStaticData->ruleNames;
}

const std::vector<std::string>& ToyCalculatorLexer::getChannelNames() const {
  return toycalculatorlexerLexerStaticData->channelNames;
}

const std::vector<std::string>& ToyCalculatorLexer::getModeNames() const {
  return toycalculatorlexerLexerStaticData->modeNames;
}

const dfa::Vocabulary& ToyCalculatorLexer::getVocabulary() const {
  return toycalculatorlexerLexerStaticData->vocabulary;
}

antlr4::atn::SerializedATNView ToyCalculatorLexer::getSerializedATN() const {
  return toycalculatorlexerLexerStaticData->serializedATN;
}

const atn::ATN& ToyCalculatorLexer::getATN() const {
  return *toycalculatorlexerLexerStaticData->atn;
}




void ToyCalculatorLexer::initialize() {
#if ANTLR4_USE_THREAD_LOCAL_CACHE
  toycalculatorlexerLexerInitialize();
#else
  ::antlr4::internal::call_once(toycalculatorlexerLexerOnceFlag, toycalculatorlexerLexerInitialize);
#endif
}
