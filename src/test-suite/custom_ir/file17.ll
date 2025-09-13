; test2_call_volatile.ll - Function call + volatile load pattern
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

declare ptr @get_pointer()

define i32 @test_call_volatile() {
entry:
  %ptr = call ptr @get_pointer()
  %val = load volatile i32, ptr %ptr, align 4
  ret i32 %val
}