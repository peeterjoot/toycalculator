// RUN: %Not %OptSilly --source %s 2>&1 | %FileCheck %s

module {
  func.func private @foo() -> i32 {
      %0 = "silly.declare"() : () -> !silly.var<i8[20]>
      %1 = "silly.string_literal"() <{value = "A string literal!"}> : () -> !llvm.ptr
      silly.assign %0 : <i8[20]> = %1 : !llvm.ptr
      %2 = silly.load %0 : <i8[20]> : !llvm.ptr
      // CHECK: error: type of return operand 0 ('!llvm.ptr') doesn't match function result type ('i32') in function @foo
      "func.return"(%2) : (!llvm.ptr) -> ()
  }
}
