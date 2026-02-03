// RUN: %Not %OptSilly --source %s 2>&1 | %FileCheck %s

module {
  func.func private @foo() -> (i1) {
    "silly.scope"() ({
      %c3_i32 = arith.constant 3 : i32
      // CHECK: error: 'silly.return' op must be the last operation in the parent block
      "silly.return"(%c3_i32) : (i32) -> ()
      "silly.return"(%c3_i32) : (i32) -> ()
    }) : () -> ()
    "silly.yield"() : () -> ()
  }
}

