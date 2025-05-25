# gdb ./toycalculator ; source ../b

b main
#run -g --stdout ../samples/bin.toy --no-emit-object  --debug
#run -g --stdout ../samples/types.toy --no-emit-object  --debug --emit-mlir
run ../samples/test.toy --stdout --emit-mlir --no-emit-object --debug -g --emit-llvm
#run -g --stdout ../samples/unary.toy --no-emit-object  --debug --emit-mlir
#run -g --stdout ../samples/return3.toy  --debug --emit-mlir --no-emit-object
#run -g --stdout ../samples/returnx.toy  --debug --emit-mlir --no-emit-object
#run -g --stdout ../samples/unary.toy --no-emit-object  --debug

b __assert_perror_fail

#b createDICompileUnitAttr
#b createDISubprogram
b ToyToLLVMLoweringPass::runOnOperation
#b translateModuleToLLVMIR
#b DebugTranslation::DebugTranslation
#b getDbgRecordRange

#b buildUnaryExpression
#b enterAssignment
#b enterAssignmentExpression
#b enterBinaryexpression
#b enterBoolDeclare
#b enterDeclare
#b enterFloatDeclare
#b enterIntDeclare
#b enterPrint
#b enterReturn
#b enterReturnstatement
#b enterStartRule
#b enterUnaryexpression
#b exitReturnstatement
#b MLIRListener::enterAssignmentExpression
#b MLIRListener::registerDeclaration
#b ProgramOpLowering::matchAndRewrite
#b ReturnOpLowering::matchAndRewrite
#b toy::AssignOpLowering::matchAndRewrite
#b toy::DeclareOpLowering::matchAndRewrite
#b toy::ExitOpLowering::matchAndRewrite
#b toy::LoadOpLowering::matchAndRewrite
#b toy::MLIRListener::enterAssignment
#b toy::PrintOpLowering::matchAndRewrite
#b toy::ProgramOpLowering::matchAndRewrite
