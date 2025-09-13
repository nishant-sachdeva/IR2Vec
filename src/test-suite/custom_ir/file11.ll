; File: test1_basic_chains.ll
; Basic pointer chains and GEPs - simple case
; Compile with: clang -S -emit-llvm -O1 test1.c

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

%struct.Node = type { i32, i32, ptr }

@global_node = dso_local global %struct.Node { i32 10, i32 20, ptr null }, align 8

define dso_local i32 @main() #0 {
entry:
  %node = alloca %struct.Node, align 8
  %field1 = getelementptr inbounds %struct.Node, ptr %node, i32 0, i32 0
  store i32 42, ptr %field1, align 4
  %field2 = getelementptr inbounds %struct.Node, ptr %node, i32 0, i32 1
  %val = load i32, ptr %field1, align 4
  store i32 %val, ptr %field2, align 4
  
  ; Access global
  %global_field = getelementptr inbounds %struct.Node, ptr @global_node, i32 0, i32 0
  %global_val = load i32, ptr %global_field, align 4
  
  ret i32 %global_val
}

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
