; test5_atomic_volatile.ll - Mixed atomic and volatile operations
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

define void @test_atomic_volatile() {
entry:
  %ptr = alloca i32, align 4
  %atomic_val = load atomic i32, ptr %ptr monotonic, align 4
  %volatile_val = load volatile i32, ptr %ptr, align 4
  store atomic i32 200, ptr %ptr monotonic, align 4
  ret void
}