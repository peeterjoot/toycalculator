#java -Xmx500M -cp "/usr/local/lib/antlr-4.13.2-complete.jar:$CLASSPATH" org.antlr.v4.gui.TestRig -Dlanguage=Cpp calculator.g4 -listener

set -x

rm -rf grammar build
mkdir -p grammar

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
#calculator.g4 \
#-listener \
#-o grammar

# Download and use 4.10 explicitly so the generated grammar files match the installed runtime:
java \
-Xmx500M \
-cp `pwd`/antlr-4.10-complete.jar \
org.antlr.v4.Tool \
-Dlanguage=Cpp \
calculator.g4 \
-listener \
-o grammar


