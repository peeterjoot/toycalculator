# This is a gdb script.  Use it like so from samples/
#
# gdb -q ../build/toycalculator ; source ../b
#
# (bin/debugit does this)

b main
# pick which sample program to use for the compiler debugging.
#run ../samples/function_plist.toy --stdout --emit-mlir --no-emit-object --debug -g --emit-llvm
run ../samples/call_with_param_referenced.toy --stdout --emit-mlir --no-emit-object --debug -g --emit-llvm

b __assert_perror_fail
b createPrintCall
#b lookupLocalSymbolReference
#b createLocalSymbolReference
#b lookupDeclareForVar
#b MLIRListener::enterDeclare
#b MLIRListener::enterBoolDeclare
#b MLIRListener::enterIntDeclare
#b MLIRListener::enterFunction
#b MLIRListener::exitFunction
#b MLIRListener::enterFloatDeclare
#b MLIRListener::enterStringDeclare
#b MLIRListener::enterIfelifelse
#b MLIRListener::enterPrint
#b MLIRListener::enterReturnStatement
#b MLIRListener::enterExitStatement
#b MLIRListener::enterAssignment
#b MLIRListener::enterAssignmentExpression

#b MLIRListener::processReturnLike
#b createPrintCall
#b enterPrint

#b toy::DeclareOpLowering::matchAndRewrite
#b MLIRListener::createScope
#b ScopeOpLowering::matchAndRewrite

#b toy::MLIRListener::enterCall
#b toy::MLIRListener::enterFunction

#b buildUnaryExpression
#b createDICompileUnit
#b createDICompileUnitAttr
#b createDISubprogram
#b createFuncDebug
#b createLocalSymbolReference
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
#b lookupAllocaForVar
#b lookupDeclareForVar
#b lookupFuncNameForOp
#b lookupLocalSymbolReference
#b mlir::createToyToLLVMLoweringPass
#b MLIRListener::enterAssignmentExpression
#b MLIRListener::enterStartRule
#b MLIRListener::registerDeclaration
#b ProgramOpLowering::matchAndRewrite
#b ReturnOpLowering::matchAndRewrite
#b toy::AssignOpLowering::matchAndRewrite
#b toy::DeclareOpLowering::matchAndRewrite
#b toy::ExitOpLowering::matchAndRewrite
#b toy::FuncOp::addEntryBlock
#b toy::FuncOpLowering::matchAndRewrite
#b toy::LoadOpLowering::matchAndRewrite
#b toy::MLIRListener::enterAssignment
#b toy::MLIRListener::lookupDeclareForVar
#b toy::PrintOpLowering::matchAndRewrite
#b toy::ProgramOpLowering::matchAndRewrite
#b ToyToLLVMLoweringPass::runOnOperation
#b translateModuleToLLVMIR
