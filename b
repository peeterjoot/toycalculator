# This is a gdb script.  Use it like so from samples/
#
# gdb -q ../build/toycalculator ; source ../b
#
# (bin/debugit does this)

b main
# pick which sample program to use for the compiler debugging.
#run ../samples/test.toy --stdout --emit-mlir --no-emit-object --debug -g --emit-llvm
run ../samples/dcl_assign.toy --stdout --emit-mlir --no-emit-object --debug -g --emit-llvm

#b lookupDeclareForVar
#b lookupAllocaForVar

#b driver.cpp:277
#b mlir::createToyToLLVMLoweringPass
#b toy::MLIRListener::lookupDeclareForVar
#b toy::FuncOp::addEntryBlock
#b MLIRListener::enterStartRule
b __assert_perror_fail
#b parser.cpp:597
#b MLIRListener::enterAssignmentExpression

#b buildUnaryExpression
#b createDICompileUnitAttr
#b createDISubprogram
#b DebugTranslation::DebugTranslation
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
#b getDbgRecordRange
#b MLIRListener::enterAssignmentExpression
#b MLIRListener::registerDeclaration
#b parser.cpp:131
#b ProgramOpLowering::matchAndRewrite
#b ReturnOpLowering::matchAndRewrite
#b std::stoi
#b toy::AssignOpLowering::matchAndRewrite
b toy::FuncOpLowering::matchAndRewrite
#b toy::DeclareOpLowering::matchAndRewrite
#b toy::ExitOpLowering::matchAndRewrite
#b toy::LoadOpLowering::matchAndRewrite
#b toy::MLIRListener::enterAssignment
#b toy::PrintOpLowering::matchAndRewrite
#b toy::ProgramOpLowering::matchAndRewrite
#b ToyToLLVMLoweringPass::runOnOperation
#b translateModuleToLLVMIR
