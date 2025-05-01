#b enterStartRule
#b enterDeclare
#b enterPrint
#b enterAssignment
#b enterUnaryexpression
#b enterBinaryexpression
#b enterReturn
b __assert_perror_fail

#b createDICompileUnitAttr
#b createDISubprogram
#b ToyToLLVMLoweringPass::runOnOperation
#b translateModuleToLLVMIR
#b DebugTranslation::DebugTranslation
b driver.cpp:284
b getDbgRecordRange

#b ProgramOpLowering::matchAndRewrite
#b ReturnOpLowering::matchAndRewrite
#b driver.cpp:114

## firstError assignments:
#b driver.cpp:149
#b driver.cpp:167
#b driver.cpp:186
## throw point:
#b driver.cpp:137

run -g --stdout ../samples/foo.toy --no-emit-object  --debug
