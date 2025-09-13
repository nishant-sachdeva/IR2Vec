
; File: test3_phi_select.ll
; PHI nodes and select instructions with memory ops

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

define dso_local i32 @test_phi_memory(i32 noundef %flag) #0 {
entry:
  %ptr1 = alloca i32, align 4
  %ptr2 = alloca i32, align 4
  %ptr3 = alloca i32, align 4
  
  store i32 10, ptr %ptr1, align 4
  store i32 20, ptr %ptr2, align 4
  store i32 30, ptr %ptr3, align 4
  
  %cmp = icmp sgt i32 %flag, 5
  br i1 %cmp, label %then, label %else

then:
  %val1 = load i32, ptr %ptr1, align 4
  %new_val1 = add nsw i32 %val1, 100
  store i32 %new_val1, ptr %ptr1, align 4
  br label %merge

else:
  %val2 = load i32, ptr %ptr2, align 4
  %new_val2 = add nsw i32 %val2, 200
  store i32 %new_val2, ptr %ptr2, align 4
  br label %merge

merge:
  %phi_ptr = phi ptr [ %ptr1, %then ], [ %ptr2, %else ]
  %phi_val = phi i32 [ %new_val1, %then ], [ %new_val2, %else ]
  
  ; Use select with the phi values
  %cmp2 = icmp sgt i32 %phi_val, 100
  %selected_ptr = select i1 %cmp2, ptr %phi_ptr, ptr %ptr3
  
  %final_val = load i32, ptr %selected_ptr, align 4
  %result = add nsw i32 %final_val, %phi_val
  store i32 %result, ptr %selected_ptr, align 4
  
  ret i32 %result
}

define dso_local i32 @main() #0 {
entry:
  %result1 = call i32 @test_phi_memory(i32 noundef 10)
  %result2 = call i32 @test_phi_memory(i32 noundef 2)
  %final = add nsw i32 %result1, %result2
  ret i32 %final
}

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
