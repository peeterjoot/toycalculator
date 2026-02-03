// RUN: %Not %OptSilly --source %s 2>&1 | %FileCheck %s

module {
  // CHECK: error: use of undeclared SSA value name
  %0 = "silly.declare"(%c1_i64) <{sym_name = "anInitializedScalar"}> : (i64) -> !silly.var<i64>
}
