// RUN: %Not %OptSilly --source %s 2>&1 | %FileCheck %s

module {
  func.func private @foo() -> (i32) {
    "silly.scope"() ({
      // CHECK: error: 'silly.return' op must return exactly one value when function has a return type
      "silly.return"() : () -> ()
    }) : () -> ()
    "silly.yield"() : () -> ()
  }
}
