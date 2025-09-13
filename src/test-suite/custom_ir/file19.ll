; test4_alloca_gep.ll - Alloca + GEP + volatile load
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

%struct.TestStruct = type { i32, i16 }

define void @test_alloca_gep_volatile() {
entry:
  %local = alloca %struct.TestStruct, align 8
  %field1 = getelementptr inbounds %struct.TestStruct, ptr %local, i32 0, i32 0
  %field2 = getelementptr inbounds %struct.TestStruct, ptr %local, i32 0, i32 1
  store i32 100, ptr %field1, align 4
  %val1 = load i32, ptr %field1, align 4
  %val2 = load volatile i16, ptr %field2, align 2
  ret void
}