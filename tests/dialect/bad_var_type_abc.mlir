// RUN: %Not %OptSilly --source %s 2>&1 | %FileCheck %s

module {
  // CHECK: error: expected integer value
  // CHECK: error: array-size must be an integer
  %bad = "silly.declare"() : () -> !silly.var<i64[abc]>
}
