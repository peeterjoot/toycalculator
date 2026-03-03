// RUN: %Not %ExeSilly -c %s 2>&1 | %FileCheck %s
// CHECK: silly: error: Failed to parse MLIR file

module {
  func.func private @identity(%arg0: i16) -> i16 {
    xreturn %arg0 : i16
  }
}
