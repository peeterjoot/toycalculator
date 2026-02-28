# was: tests/endtoend/simple/simple/manual_command_line_options_test.sh

rm -f *.o *.mlir *.mlirbc *.ll simpleless
silly --emit-mlir simpleless.silly
ls *.o *.mlir *.mlirbc *.ll simpleless
file *.mlir
head *.mlir

# expect no .o, a .mlir (text), no .mlirbc, no .ll

rm -f *.o *.mlir *.mlirbc *.ll simpleless
silly -c --emit-mlir simpleless.silly
ls *.o *.mlir *.mlirbc *.ll simpleless
file *.mlir
cat *.mlir | head

# expect no .o, a .mlir (text), no .mlirbc, no .ll, no simpleless

rm -f *.o *.mlir *.mlirbc *.ll simpleless
silly -c --emit-mlirbc simpleless.silly
ls *.o *.mlir *.mlirbc *.ll simpleless
file *.mlirbc
silly-opt -s *.mlirbc | head

# expect no .o, a .mlirbc (binary), no .mlir, no .ll, no simpleless

rm -f *.o *.mlir *.mlirbc *.ll simpleless
silly -c simpleless.silly
ls *.o *.mlir *.mlirbc *.ll simpleless
file *.o
objdump -d *.o | head

# expect .o, no .mlir (text), no .mlirbc, no .ll, no simpleless
