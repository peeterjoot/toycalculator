// RUN: %Not %OptSilly --source %s 2>&1 | %FileCheck %s

// silly.return directly inside func.func body (no enclosing silly.scope)
module {
  %c3_i32 = arith.constant 3 : i32
  // CHECK: error: 'silly.return' op must appear inside a 'silly.scope' block
  "silly.return"(%c3_i32) : (i32) -> ()
}
