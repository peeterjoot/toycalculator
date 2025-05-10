# gdb ./toycalculator ; source ../b

b main
#run -g --stdout ../samples/bin.toy --no-emit-object  --debug
#run -g --stdout ../samples/types.toy --no-emit-object  --debug --emit-mlir
run ../samples/bool.toy --stdout --emit-mlir --no-emit-object --debug
#run -g --stdout ../samples/unary.toy --no-emit-object  --debug --emit-mlir
#run -g --stdout ../samples/return3.toy  --debug --emit-mlir --no-emit-object
#run -g --stdout ../samples/returnx.toy  --debug --emit-mlir --no-emit-object
#b driver.cpp:194
#run -g --stdout ../samples/unary.toy --no-emit-object  --debug

#b ToyParser.cpp:167

#b buildUnaryExpression
b DeclareOpLowering::matchAndRewrite
b ProgramOpLowering::matchAndRewrite
#b lowering.cpp:301
b AssignOpLowering::matchAndRewrite
#b ExitOpLowering::matchAndRewrite

#b ToyToLLVMLowering.cpp:455
#b enterStartRule
#b enterDeclare
#b enterBoolDeclare
#b enterIntDeclare
#b enterFloatDeclare
#b enterPrint
#b enterAssignment
#b enterAssignmentExpression
#b MLIRListener::registerDeclaration
#b enterUnaryexpression
#b enterBinaryexpression
#b enterReturn
#b enterReturnstatement
#b ReturnOpLowering::matchAndRewrite
#b exitReturnstatement
b __assert_perror_fail
#b _ZN3toy12MLIRListener20enterReturnstatementEPN9ToyParser22ReturnstatementContextE

#b createDICompileUnitAttr
#b createDISubprogram
#b ToyToLLVMLoweringPass::runOnOperation
#b translateModuleToLLVMIR
#b DebugTranslation::DebugTranslation
#b driver.cpp:284
#b getDbgRecordRange

#b toy::MLIRListener::enterAssignment

#b ProgramOpLowering::matchAndRewrite
#b ReturnOpLowering::matchAndRewrite
#b driver.cpp:114

## firstError assignments:
#b driver.cpp:149
#b driver.cpp:167
#b driver.cpp:186
## throw point:
#b driver.cpp:137

