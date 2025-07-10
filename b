# This is a gdb script.  Use it like so from samples/
#
# gdb -q ../build/toycalculator ; source ../b
#
# (bin/debugit does this)

b main
# pick which sample program to use for the compiler debugging.
#run ../samples/function_plist.toy --stdout --emit-mlir --no-emit-object --debug -g --emit-llvm
#run ../samples/stringlit.toy --stdout --emit-mlir --no-emit-object --debug -g --emit-llvm
run ../samples/function_return.toy --stdout --emit-mlir --no-emit-object --debug -g --emit-llvm
#run ../samples/bool.toy --stdout --emit-mlir --no-emit-object --debug -g --emit-llvm
#run ../samples/function_void_intparm.toy --stdout --emit-mlir --no-emit-object --debug -g --emit-llvm

b __assert_perror_fail

b markExplicitTerminator
b wasTerminatorExplicit

#b buildUnaryExpression
#b createDICompileUnit
#b createDICompileUnitAttr
#b createDISubprogram
#b createFuncDebug
#b createLocalSymbolReference
#b createPrintCall
#b DebugTranslation::DebugTranslation
#b enterBinaryexpression
#b enterUnaryexpression
b exitReturnstatement
#b getDbgRecordRange
#b lookupAllocaForVar
#b lookupDeclareForVar
#b lookupFuncNameForOp
#b lookupLocalSymbolReference
#b mlir::createToyToLLVMLoweringPass
#b ProgramOpLowering::matchAndRewrite
#b ReturnOpLowering::matchAndRewrite
#b ScopeOpLowering::matchAndRewrite
#b toy::AssignOpLowering::matchAndRewrite
#b toy::DeclareOpLowering::matchAndRewrite
#b toy::ExitOpLowering::matchAndRewrite
#b toy::FuncOp::addEntryBlock
#b toy::FuncOpLowering::matchAndRewrite
#b toy::LoadOpLowering::matchAndRewrite
#b toy::MLIRListener::createScope
#b toy::MLIRListener::enterAssignment
#b toy::MLIRListener::enterAssignmentExpression
#b toy::MLIRListener::enterBoolDeclare
#b toy::MLIRListener::enterCall
#b toy::MLIRListener::enterDeclare
#b toy::MLIRListener::enterExitStatement
#b toy::MLIRListener::enterExitStatement 
#b toy::MLIRListener::enterFloatDeclare
#b toy::MLIRListener::enterFunction
#b toy::MLIRListener::enterIfelifelse
#b toy::MLIRListener::enterIntDeclare
#b toy::MLIRListener::enterPrint
b toy::MLIRListener::enterReturnStatement
#b toy::MLIRListener::enterStartRule
#b toy::MLIRListener::enterStringDeclare
#b toy::MLIRListener::exitFunction
#b toy::MLIRListener::exitStartRule 
#b toy::MLIRListener::lookupDeclareForVar
#b toy::MLIRListener::processReturnLike
#b toy::MLIRListener::registerDeclaration
#b toy::PrintOpLowering::matchAndRewrite
#b toy::ProgramOpLowering::matchAndRewrite
#b ToyToLLVMLoweringPass::runOnOperation
#b translateModuleToLLVMIR
