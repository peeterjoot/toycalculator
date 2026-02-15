// RUN: %Not %OptSilly --source %s 2>&1 | %FileCheck %s

module {
  func.func private @foo() -> (!silly.var<i32[7]>) {
    "silly.scope"() ({
      %0 = "silly.declare"() : () -> !silly.var<i32[7]>
      // CHECK: 'silly.return' op function return type must be scalar (integer or floating-point), got '!silly.var<i32[7]>'
      "silly.return"(%0) : (!silly.var<i32[7]>) -> ()
    }) : () -> ()
    "silly.yield"() : () -> ()
  }
}

