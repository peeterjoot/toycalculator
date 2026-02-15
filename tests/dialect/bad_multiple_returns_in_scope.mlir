// RUN: %Not %OptSilly --source %s 2>&1 | %FileCheck %s

module {
  func.func private @foo() -> (i1) {
      %c3_i32 = arith.constant 3 : i32
      // CHECK: error: 'func.return' op must be the last operation in the parent block
      "func.return"(%c3_i32) : (i32) -> ()
      "func.return"(%c3_i32) : (i32) -> ()
  }
}

