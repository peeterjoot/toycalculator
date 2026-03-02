; RUN: %Not %ExeSilly %s 2>&1 | %FileCheck %s --check-prefix=ERR
;
; ERR: silly: error: Failed to translate to LLVM IR or parse supplied LLVM IR

define i16 @addone(i16 %0) {
  xret i16 %0
}
