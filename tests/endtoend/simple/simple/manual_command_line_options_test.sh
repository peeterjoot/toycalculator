rm -f *.o *.mlir *.mlirbc *.ll simpleless
silly --emit-mlir simpleless.silly
ls *.o *.mlir *.mlirbc *.ll simpleless
file *.mlir
head *.mlir

# expect no .o, a .mlir (text), no .mlirbc, no .ll

rm -f *.o *.mlir *.mlirbc *.ll simpleless
silly -c --emit-mlir simpleless.silly
ls *.o *.mlir *.mlirbc *.ll simpleless
file *.mlirbc
silly-opt -s *.mlirbc | head

# expect no .o, a .mlirbc (binary), no .mlir, no .ll, no simpleless

rm -f *.o *.mlir *.mlirbc *.ll simpleless
silly -S simpleless.silly
ls *.o *.mlir *.mlirbc *.ll simpleless
file *.mlir
head *.mlir

# expect no .o, a .mlir (text), no .mlirbc, no .ll, no simpleless

rm -f *.o *.mlir *.mlirbc *.ll simpleless
silly -c simpleless.silly
ls *.o *.mlir *.mlirbc *.ll simpleless
file *.o
objdump -d *.o | head

# expect .o, no .mlir (text), no .mlirbc, no .ll, no simpleless

rm -f *.o *.mlir *.mlirbc *.ll simpleless
silly -S -c simpleless.silly
echo $?
ls *.o *.mlir *.mlirbc *.ll simpleless

# expect no outputs, only failure
