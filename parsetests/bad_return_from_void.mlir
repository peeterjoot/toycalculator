// RUN: %Not %OptSilly --source %s 2>&1 | %FileCheck %s

module {
  func.func private @foo() -> () {
    "silly.scope"() ({
      %c3_i32 = arith.constant 3 : i32
      // CHECK: error: 'silly.return' op cannot return a value because enclosing function has no return type (void)
      "silly.return"(%c3_i32) : (i32) -> ()
    }) : () -> ()
    "silly.yield"() : () -> ()
  }
}
