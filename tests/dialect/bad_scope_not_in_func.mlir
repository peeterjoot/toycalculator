// RUN: %Not %OptSilly --source %s 2>&1 | %FileCheck %s

module {
  // CHECK: error: 'silly.scope' op silly.scope must be inside a 'func.func'
  "silly.scope"() ({
  }) : () -> ()
}
