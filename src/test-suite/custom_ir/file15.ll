; File: test5_loops_chaos.ll
; Ultimate chaos: loops, dynamic allocation, nested structures

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

%struct.ListNode = type { i32, ptr, ptr }

declare dso_local noalias ptr @malloc(i64 noundef) #1
declare dso_local void @free(ptr noundef) #1
declare dso_local i32 @rand() #1

define dso_local ptr @create_list(i32 noundef %size) #0 {
entry:
  %cmp = icmp sle i32 %size, 0
  br i1 %cmp, label %cleanup, label %for.cond.preheader

for.cond.preheader:
  br label %for.body

for.body:
  %i.05 = phi i32 [ 0, %for.cond.preheader ], [ %inc, %for.inc ]
  %current.04 = phi ptr [ null, %for.cond.preheader ], [ %call, %for.inc ]
  %head.03 = phi ptr [ null, %for.cond.preheader ], [ %head.1, %for.inc ]
  
  %call = call noalias ptr @malloc(i64 noundef 24)
  %tobool = icmp ne ptr %call, null
  br i1 %tobool, label %if.then, label %for.inc

if.then:
  %call1 = call i32 @rand()
  %rem = srem i32 %call1, 1000
  %data = getelementptr inbounds %struct.ListNode, ptr %call, i32 0, i32 0
  store i32 %rem, ptr %data, align 8
  
  %next = getelementptr inbounds %struct.ListNode, ptr %call, i32 0, i32 1
  store ptr null, ptr %next, align 8
  
  %prev = getelementptr inbounds %struct.ListNode, ptr %call, i32 0, i32 2
  store ptr %current.04, ptr %prev, align 8
  
  %tobool2 = icmp ne ptr %current.04, null
  br i1 %tobool2, label %if.then3, label %if.else

if.then3:
  %next4 = getelementptr inbounds %struct.ListNode, ptr %current.04, i32 0, i32 1
  store ptr %call, ptr %next4, align 8
  br label %for.inc

if.else:
  br label %for.inc

for.inc:
  %head.1 = phi ptr [ %head.03, %if.then3 ], [ %call, %if.else ], [ %head.03, %for.body ]
  %inc = add nsw i32 %i.05, 1
  %cmp5 = icmp slt i32 %inc, %size
  br i1 %cmp5, label %for.body, label %cleanup

cleanup:
  %retval.0 = phi ptr [ null, %entry ], [ %head.1, %for.inc ]
  ret ptr %retval.0
}

define dso_local i32 @process_list_chaotically(ptr noundef %head, i32 noundef %mode) #0 {
entry:
  %tobool = icmp ne ptr %head, null
  br i1 %tobool, label %while.body.preheader, label %while.end

while.body.preheader:
  br label %while.body

while.body:
  %current.018 = phi ptr [ %current.1, %while.continue ], [ %head, %while.body.preheader ]
  %sum.017 = phi i32 [ %sum.2, %while.continue ], [ 0, %while.body.preheader ]
  %count.016 = phi i32 [ %inc8, %while.continue ], [ 0, %while.body.preheader ]
  
  %data = getelementptr inbounds %struct.ListNode, ptr %current.018, i32 0, i32 0
  %val = load i32, ptr %data, align 8
  
  %cmp = icmp eq i32 %mode, 1
  br i1 %cmp, label %if.then, label %if.else

if.then:
  %mul = mul nsw i32 %val, 2
  store i32 %mul, ptr %data, align 8
  %add = add nsw i32 %sum.017, %mul
  br label %if.end6

if.else:
  %cmp1 = icmp eq i32 %mode, 2
  br i1 %cmp1, label %if.then2, label %if.else4

if.then2:
  %prev = getelementptr inbounds %struct.ListNode, ptr %current.018, i32 0, i32 2
  %prev_node = load ptr, ptr %prev, align 8
  %tobool3 = icmp ne ptr %prev_node, null
  br i1 %tobool3, label %land.lhs.true, label %if.else4

land.lhs.true:
  %prev_data = getelementptr inbounds %struct.ListNode, ptr %prev_node, i32 0, i32 0
  %prev_val = load i32, ptr %prev_data, align 8
  %combined = add nsw i32 %val, %prev_val
  store i32 %combined, ptr %data, align 8
  %add3 = add nsw i32 %sum.017, %combined
  br label %if.end6

if.else4:
  %sub = sub nsw i32 %val, %count.016
  store i32 %sub, ptr %data, align 8
  %add5 = add nsw i32 %sum.017, %sub
  br label %if.end6

if.end6:
  %sum.1 = phi i32 [ %add, %if.then ], [ %add3, %land.lhs.true ], [ %add5, %if.else4 ]
  
  %rem = srem i32 %count.016, 3
  %cmp7 = icmp eq i32 %rem, 0
  br i1 %cmp7, label %if.then8, label %while.continue

if.then8:
  %next = getelementptr inbounds %struct.ListNode, ptr %current.018, i32 0, i32 1
  %next_node = load ptr, ptr %next, align 8
  %tobool9 = icmp ne ptr %next_node, null
  %spec.select = select i1 %tobool9, ptr %next_node, ptr %current.018
  %new_val = add nsw i32 %sum.1, 100
  %target_data = getelementptr inbounds %struct.ListNode, ptr %spec.select, i32 0, i32 0
  store i32 %new_val, ptr %target_data, align 8
  %sum.2 = add nsw i32 %sum.1, %new_val
  br label %while.continue

while.continue:
  %sum.3 = phi i32 [ %sum.2, %if.then8 ], [ %sum.1, %if.end6 ]
  %inc8 = add nsw i32 %count.016, 1
  %next10 = getelementptr inbounds %struct.ListNode, ptr %current.018, i32 0, i32 1
  %current.1 = load ptr, ptr %next10, align 8
  %tobool11 = icmp ne ptr %current.1, null
  %cmp12 = icmp slt i32 %inc8, 20
  %and = and i1 %tobool11, %cmp12
  br i1 %and, label %while.body, label %while.end

while.end:
  %sum.0.lcssa = phi i32 [ 0, %entry ], [ %sum.3, %while.continue ]
  ret i32 %sum.0.lcssa
}

define dso_local void @free_list(ptr noundef %head) #0 {
entry:
  br label %while.cond

while.cond:
  %current.0 = phi ptr [ %head, %entry ], [ %next, %while.body ]
  %tobool = icmp ne ptr %current.0, null
  br i1 %tobool, label %while.body, label %while.end

while.body:
  %next.ptr = getelementptr inbounds %struct.ListNode, ptr %current.0, i32 0, i32 1
  %next = load ptr, ptr %next.ptr, align 8
  call void @free(ptr noundef %current.0)
  br label %while.cond

while.end:
  ret void
}

define dso_local i32 @main() #0 {
entry:
  %list1 = call ptr @create_list(i32 noundef 10)
  %list2 = call ptr @create_list(i32 noundef 5)
  
  %result1 = call i32 @process_list_chaotically(ptr noundef %list1, i32 noundef 1)
  %result2 = call i32 @process_list_chaotically(ptr noundef %list2, i32 noundef 2)
  %result3 = call i32 @process_list_chaotically(ptr noundef %list1, i32 noundef 3)
  
  call void @free_list(ptr noundef %list1)
  call void @free_list(ptr noundef %list2)
  
  %sum = add nsw i32 %result1, %result2
  %final = add nsw i32 %sum, %result3
  ret i32 %final
}

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }