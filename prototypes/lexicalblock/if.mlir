module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i8 = dense<[8, 32]> : vector<2xi64>, i16 = dense<[16, 32]> : vector<2xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 32, 64>, "dlti.stack_alignment" = 128 : i64, "dlti.function_pointer_alignment" = #dlti.function_pointer_alignment<32, function_dependent = true>>, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32", llvm.ident = "flang version 22.1.1 (https://github.com/llvm/llvm-project.git fef02d48c08db859ef83f84232ed78bd9d1c323a)", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @_QQmain() attributes {fir.bindc_name = "IFSCOPE"} {
    %false = arith.constant false
    %c10_i32 = arith.constant 10 : i32
    %c6_i32 = arith.constant 6 : i32
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %c3_i32 = arith.constant 3 : i32
    %0 = fir.dummy_scope : !fir.dscope
    %1 = fir.alloca i64 {bindc_name = "myscopevar", uniq_name = "_QFEmyscopevar"}
    %2 = fir.declare %1 {uniq_name = "_QFEmyscopevar"} : (!fir.ref<i64>) -> !fir.ref<i64>
    %3 = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFEx"}
    %4 = fir.declare %3 {uniq_name = "_QFEx"} : (!fir.ref<i32>) -> !fir.ref<i32>
    fir.store %c3_i32 to %4 : !fir.ref<i32>
    %5 = fir.load %4 : !fir.ref<i32>
    %6 = arith.cmpi slt, %5, %c4_i32 : i32
    fir.if %6 {
      %8 = fir.load %4 : !fir.ref<i32>
      %9 = arith.addi %8, %c1_i32 : i32
      %10 = fir.convert %9 : (i32) -> i64
      fir.store %10 to %2 : !fir.ref<i64>
      %11 = fir.address_of(@_QQclXde57972bd633c25531337e418c543f78) : !fir.ref<!fir.char<1,46>>
      %12 = fir.convert %11 : (!fir.ref<!fir.char<1,46>>) -> !fir.ref<i8>
      %13 = fir.call @_FortranAioBeginExternalListOutput(%c6_i32, %12, %c10_i32) fastmath<contract> : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
      %14 = fir.load %2 : !fir.ref<i64>
      %15 = fir.call @_FortranAioOutputInteger64(%13, %14) fastmath<contract> : (!fir.ref<i8>, i64) -> i1
      %16 = fir.call @_FortranAioEndIoStatement(%13) fastmath<contract> : (!fir.ref<i8>) -> i32
    }
    %7 = fir.load %4 : !fir.ref<i32>
    fir.call @_FortranAStopStatement(%7, %false, %false) fastmath<contract> : (i32, i1, i1) -> ()
    fir.unreachable
  }
  func.func private @_FortranAioBeginExternalListOutput(i32, !fir.ref<i8>, i32) -> !fir.ref<i8> attributes {fir.io, fir.runtime}
  fir.global linkonce @_QQclXde57972bd633c25531337e418c543f78 constant : !fir.char<1,46> {
    %0 = fir.string_lit "/home/peeter/toycalculator/tests/debug/if.f90\00"(46) : !fir.char<1,46>
    fir.has_value %0 : !fir.char<1,46>
  }
  func.func private @_FortranAioOutputInteger64(!fir.ref<i8>, i64) -> i1 attributes {fir.io, fir.runtime}
  func.func private @_FortranAioEndIoStatement(!fir.ref<i8>) -> i32 attributes {fir.io, fir.runtime}
  func.func private @_FortranAStopStatement(i32, i1, i1) attributes {fir.runtime}
  func.func private @_FortranAProgramStart(i32, !llvm.ptr, !llvm.ptr, !llvm.ptr)
  func.func private @_FortranAProgramEndStatement()
  func.func @main(%arg0: i32, %arg1: !llvm.ptr, %arg2: !llvm.ptr) -> i32 {
    %c0_i32 = arith.constant 0 : i32
    %0 = fir.zero_bits !fir.ref<tuple<i32, !fir.ref<!fir.array<0xtuple<!fir.ref<i8>, !fir.ref<i8>>>>>>
    fir.call @_FortranAProgramStart(%arg0, %arg1, %arg2, %0) fastmath<contract> : (i32, !llvm.ptr, !llvm.ptr, !fir.ref<tuple<i32, !fir.ref<!fir.array<0xtuple<!fir.ref<i8>, !fir.ref<i8>>>>>>) -> ()
    fir.call @_QQmain() fastmath<contract> : () -> ()
    fir.call @_FortranAProgramEndStatement() fastmath<contract> : () -> ()
    return %c0_i32 : i32
  }
}
