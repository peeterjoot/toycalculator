// RUN: %OptSilly --source %s --out /dev/null

module {
  func.func private @foo() -> i32 {
    "silly.scope"() ({
      %c3_i32 = arith.constant 3 : i32
      "silly.return"(%c3_i32) : (i32) -> ()
    }) : () -> ()
    "silly.yield"() : () -> ()
  }
}
