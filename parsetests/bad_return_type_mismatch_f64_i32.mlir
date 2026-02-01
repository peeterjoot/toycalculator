// RUN: %Not %OptSilly --source %s 2>&1 | %FileCheck %s

module {
  func.func private @foo() -> i32 {
    "silly.scope"() ({
      %cst = arith.constant 2.710000e+00 : f32
      // CHECK: error: 'silly.return' op return operand type ('f32') does not match function return type ('i32')
      "silly.return"(%cst) : (f32) -> ()
    }) : () -> ()
    "silly.yield"() : () -> ()
  }
}

