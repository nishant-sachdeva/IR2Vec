; File: test2_nested_arrays.ll
; Nested arrays and complex GEPs
; This tests deeply nested pointer arithmetic

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

@global_array = dso_local global [3 x [4 x [5 x i32]]] zeroinitializer, align 16

define dso_local i32 @test_nested_arrays() #0 {
entry:
  %local_array = alloca [2 x [3 x i32]], align 4
  
  ; Complex nested access
  %ptr1 = getelementptr inbounds [2 x [3 x i32]], ptr %local_array, i64 0, i64 1, i64 2
  store i32 100, ptr %ptr1, align 4
  
  %ptr2 = getelementptr inbounds [2 x [3 x i32]], ptr %local_array, i64 0, i64 0, i64 1
  %val1 = load i32, ptr %ptr1, align 4
  store i32 %val1, ptr %ptr2, align 4
  
  ; Global array access
  %global_ptr1 = getelementptr inbounds [3 x [4 x [5 x i32]]], ptr @global_array, i64 0, i64 1, i64 2, i64 3
  store i32 200, ptr %global_ptr1, align 4
  
  %global_ptr2 = getelementptr inbounds [3 x [4 x [5 x i32]]], ptr @global_array, i64 0, i64 0, i64 1, i64 2
  %global_val = load i32, ptr %global_ptr1, align 4
  store i32 %global_val, ptr %global_ptr2, align 4
  
  %result = load i32, ptr %ptr2, align 4
  ret i32 %result
}

define dso_local i32 @main() #0 {
entry:
  %result = call i32 @test_nested_arrays()
  ret i32 %result
}

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
