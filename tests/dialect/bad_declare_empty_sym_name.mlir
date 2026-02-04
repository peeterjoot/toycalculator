// RUN: %Not %OptSilly --source %s 2>&1 | %FileCheck %s

module {
  // CHECK: error: 'silly.declare' op requires a non-empty 'sym_name' attribute of type StringAttr.
  %1 = "silly.declare"() <{sym_name = ""}> : () -> !silly.var<i64>
}
