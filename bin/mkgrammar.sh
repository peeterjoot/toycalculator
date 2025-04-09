#java -Xmx500M -cp "/usr/local/lib/antlr-4.13.2-complete.jar:$CLASSPATH" org.antlr.v4.gui.TestRig -Dlanguage=Cpp ToyCalculator.g4 -listener

#set -x

rm -rf ToyCalculatorParser build
mkdir -p ToyCalculatorParser

# to use this version, we'd need the matching runtime
#-cp "`pwd`/antlr-4.13.2-complete.jar:$CLASSPATH" \

# ubuntu 24.04's antlr4 (4.9) package doesn't appear to be in synch with libantlr4-runtime-dev (4.10).
#java \
#-Xmx500M \
#-cp /usr/share/java/antlr4.jar:\
#/usr/share/java/antlr4-runtime.jar:\
#/usr/share/java/antlr3-runtime.jar:\
#/usr/share/java/stringtemplate4.jar \
#org.antlr.v4.Tool \
#-Dlanguage=Cpp \
#ToyCalculator.g4 \
#-listener \
#-o ToyCalculatorParser

if grep -qi "Microsoft" /proc/version || uname -r | grep -qi "Microsoft"; then

# Downloaded and use 4.10 explicitly so the generated ToyCalculatorParser files match the installed runtime:
java \
-Xmx500M \
-cp `pwd`/antlr-4.10-complete.jar \
org.antlr.v4.Tool \
-Dlanguage=Cpp \
ToyCalculator.g4 \
-listener \
-o ToyCalculatorParser

else

#ANTLR4_PREFIX=${HOME}/antlr4/usr/local
ANTLR4=${HOME}/.local/bin/antlr4

#CXXFLAGS += -I$(ANTLR4_PREFIX)/include/antlr4-runtime
#LDFLAGS += -L$(ANTLR4_PREFIX)/lib64 -Wl,-rpath,$(ANTLR4_PREFIX)/lib64
#LOADLIBES += -lantlr4-runtime

set -x

${ANTLR4} -Dlanguage=Cpp ToyCalculator.g4 -listener -o ToyCalculatorParser

fi
