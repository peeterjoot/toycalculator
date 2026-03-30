; ModuleID = 'if-hacked.mlir'
source_filename = "if-hacked.mlir"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

declare void @__silly_print(i32, ptr)

define i32 @main() !dbg !5 {
  %1 = alloca { i32, i32, i64, ptr }, i64 1, align 8
  %2 = alloca i32, i64 1, align 4
    #dbg_declare(ptr %2, !9, !DIExpression(), !11)
  %a11 = alloca i32, i64 1, align 4
    #dbg_declare(ptr %a11, !16, !DIExpression(), !19)
  store i32 3, ptr %2, align 4, !dbg !11
  %3 = load i32, ptr %2, align 4, !dbg !12
  %4 = sext i32 %3 to i64, !dbg !12
  %5 = icmp slt i64 %4, 4, !dbg !12
  br i1 %5, label %6, label %17, !dbg !13

6:                                                ; preds = %0
  %7 = load i32, ptr %2, align 4, !dbg !14
  %8 = zext i32 %7 to i64, !dbg !15
  %9 = add i64 1, %8, !dbg !15
  %10 = trunc i64 %9 to i32, !dbg !15
  store i32 %10, ptr %a11, align 4, !dbg !20
  %12 = load i32, ptr %a11, align 4, !dbg !21
  %13 = sext i32 %12 to i64, !dbg !22
  %14 = insertvalue { i32, i32, i64, ptr } { i32 1, i32 0, i64 undef, ptr undef }, i64 %13, 2, !dbg !22
  %15 = insertvalue { i32, i32, i64, ptr } %14, ptr null, 3, !dbg !22
  %16 = getelementptr { i32, i32, i64, ptr }, ptr %1, i64 0, !dbg !22
  store { i32, i32, i64, ptr } %15, ptr %16, align 8, !dbg !22
  call void @__silly_print(i32 1, ptr %1), !dbg !22
  br label %18, !dbg !13

17:                                               ; preds = %0
  br label %18, !dbg !13

18:                                               ; preds = %6, %17
  ret i32 0, !dbg !23

; uselistorder directives
  uselistorder label %18, { 1, 0 }
}

!llvm.dbg.cu = !{!0}
!llvm.ident = !{!2}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "silly", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, nameTableKind: None)
!1 = !DIFile(filename: "if.silly", directory: ".")
!2 = !{!"silly V0.11.0"}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 7, !"Dwarf Version", i32 5}
!5 = distinct !DISubprogram(name: "main", linkageName: "main", scope: !1, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
!6 = !DISubroutineType(types: !7)
!7 = !{!8}
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !DILocalVariable(name: "x", scope: !5, file: !1, line: 1, type: !10, align: 32)
!10 = !DIBasicType(name: "int32_t", size: 32, encoding: DW_ATE_signed)
!11 = !DILocation(line: 1, column: 1, scope: !5)
!12 = !DILocation(line: 3, column: 6, scope: !5)
!13 = !DILocation(line: 3, column: 1, scope: !5)
!14 = !DILocation(line: 5, column: 27, scope: !5)
!15 = !DILocation(line: 5, column: 23, scope: !5)
!16 = !DILocalVariable(name: "myScopeVar", scope: !17, file: !1, line: 5, type: !10, align: 32)
!17 = distinct !DILexicalBlock(scope: !18, file: !1, line: 5, column: 23)
!18 = distinct !DILexicalBlock(scope: !5, file: !1, line: 3, column: 6)
!19 = !DILocation(line: 5, column: 4, scope: !17)
!20 = !DILocation(line: 5, column: 4, scope: !5)
!21 = !DILocation(line: 7, column: 10, scope: !5)
!22 = !DILocation(line: 7, column: 4, scope: !5)
!23 = !DILocation(line: 9, column: 1, scope: !5)
