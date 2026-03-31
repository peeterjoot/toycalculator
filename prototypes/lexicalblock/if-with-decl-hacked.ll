; ModuleID = 'if-with-decl.silly'
source_filename = "if-with-decl.silly"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@str_0 = private constant [5 x i8] c"Done."

declare void @__silly_print(i32, ptr)

define i32 @main() !dbg !5 {
  %1 = alloca { i32, i32, i64, ptr }, i64 1, align 8
  %a7 = alloca i32, i64 1, align 4
  %2 = alloca i32, i64 1, align 4
    #dbg_declare(ptr %2, !9, !DIExpression(), !11)
  store i32 3, ptr %2, align 4, !dbg !11
  %3 = load i32, ptr %2, align 4, !dbg !12
  %4 = sext i32 %3 to i64, !dbg !12
  %5 = icmp slt i64 %4, 4, !dbg !12
  br i1 %5, label %6, label %13, !dbg !14

6:                                                ; preds = %0
    #dbg_declare(ptr %a7, !15, !DIExpression(), !17)
  store i32 42, ptr %a7, align 4, !dbg !18
  %8 = load i32, ptr %a7, align 4, !dbg !19
  %9 = sext i32 %8 to i64, !dbg !20
  %10 = insertvalue { i32, i32, i64, ptr } { i32 1, i32 0, i64 undef, ptr undef }, i64 %9, 2, !dbg !20
  %11 = insertvalue { i32, i32, i64, ptr } %10, ptr null, 3, !dbg !20
  %12 = getelementptr { i32, i32, i64, ptr }, ptr %1, i64 0, !dbg !20
  store { i32, i32, i64, ptr } %11, ptr %12, align 8, !dbg !20
  call void @__silly_print(i32 1, ptr %1), !dbg !20
  br label %14, !dbg !14

13:                                               ; preds = %0
  br label %14, !dbg !14

14:                                               ; preds = %6, %13
  %15 = getelementptr { i32, i32, i64, ptr }, ptr %1, i64 0, !dbg !21
  store { i32, i32, i64, ptr } { i32 3, i32 0, i64 5, ptr @str_0 }, ptr %15, align 8, !dbg !21
  call void @__silly_print(i32 1, ptr %1), !dbg !21
  ret i32 0, !dbg !22

; uselistorder directives
  uselistorder label %14, { 1, 0 }
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #0

attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: write) }

!llvm.dbg.cu = !{!0}
!llvm.ident = !{!2}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "silly", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, nameTableKind: None)
!1 = !DIFile(filename: "if-with-decl.silly", directory: ".")
!2 = !{!"silly V0.11.0"}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 7, !"Dwarf Version", i32 5}
!5 = distinct !DISubprogram(name: "main", linkageName: "main", scope: !1, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
!6 = !DISubroutineType(types: !7)
!7 = !{!8}
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !DILocalVariable(name: "x", scope: !5, file: !1, line: 1, type: !10, align: 32)
!10 = !DIBasicType(name: "INT32", size: 32, encoding: DW_ATE_signed)
!11 = !DILocation(line: 1, column: 1, scope: !5)
!12 = !DILocation(line: 3, column: 6, scope: !13)
!13 = distinct !DILexicalBlock(scope: !5, file: !1, line: 3, column: 1)
!14 = !DILocation(line: 3, column: 1, scope: !13)
!15 = !DILocalVariable(name: "y", scope: !16, file: !1, line: 5, type: !10, align: 32)
!16 = distinct !DILexicalBlock(scope: !13, file: !1, line: 4, column: 1)
!17 = !DILocation(line: 5, column: 3, scope: !16)
!18 = !DILocation(line: 6, column: 7, scope: !16)
!19 = !DILocation(line: 7, column: 9, scope: !16)
!20 = !DILocation(line: 7, column: 3, scope: !16)
!21 = !DILocation(line: 10, column: 1, scope: !5)
!22 = !DILocation(line: 11, column: 1, scope: !5)
