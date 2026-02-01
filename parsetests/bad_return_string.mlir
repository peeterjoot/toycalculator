// RUN: %Not %OptSilly --source %s 2>&1 | %FileCheck %s

module {
  func.func private @foo() -> i32 {
    "silly.scope"() ({
      %0 = "silly.declare"() <{sym_name = "s"}> : () -> !silly.var<i8[20]>
      %1 = "silly.string_literal"() <{value = "A string literal!"}> : () -> !llvm.ptr
      silly.assign %0 : <i8[20]> = %1 : !llvm.ptr
      %2 = silly.load %0 : <i8[20]> : !llvm.ptr
      // CHECK: error: 'silly.return' op return operand type ('!llvm.ptr') does not match function return type ('i32')
      "silly.return"(%2) : (!llvm.ptr) -> ()
    }) : () -> ()
    "silly.yield"() : () -> ()
  }
}
