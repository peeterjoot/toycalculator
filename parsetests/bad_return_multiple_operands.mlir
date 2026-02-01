// RUN: %Not %OptSilly --source %s 2>&1 | %FileCheck %s

module {
  func.func private @foo() -> i32 {
    "silly.scope"() ({
      %a = arith.constant 3 : i32
      // CHECK: error: 'silly.return' op must return exactly one value when function has a return type
      "silly.return" (%a, %a) : (i32, i32) -> ()
    }) : () -> ()
    "silly.yield"() : () -> ()
  }
}
