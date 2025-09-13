; test3_nested_calls.ll - Nested calls with volatile loads (sqlite3 pattern)
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

declare ptr @walCkptInfo(ptr)
declare ptr @walIndexHdr(ptr)

define void @test_nested_calls_volatile(ptr %input) {
entry:
  %info = call ptr @walCkptInfo(ptr %input)
  %hdr = call ptr @walIndexHdr(ptr %info)
  %val1 = load volatile i32, ptr %info, align 4
  %val2 = load volatile i32, ptr %hdr, align 4
  ret void
}