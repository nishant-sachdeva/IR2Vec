
; File: test4_complex_structs.ll
; Complex structs with nested pointers and function calls

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

%struct.ComplexNode = type { i32, ptr, [3 x ptr], %struct.Inner }
%struct.Inner = type { ptr, i64 }

@global_int = dso_local global i32 999, align 4
@global_string = dso_local global [20 x i8] c"hello world\00\00\00\00\00\00\00\00\00", align 1

declare dso_local noalias ptr @malloc(i64 noundef) #1
declare dso_local void @free(ptr noundef) #1

define dso_local ptr @create_node(i32 noundef %val) #0 {
entry:
  %call = call noalias ptr @malloc(i64 noundef 56)
  %tobool = icmp ne ptr %call, null
  br i1 %tobool, label %if.then, label %if.end

if.then:
  %value = getelementptr inbounds %struct.ComplexNode, ptr %call, i32 0, i32 0
  store i32 %val, ptr %value, align 8
  
  %next = getelementptr inbounds %struct.ComplexNode, ptr %call, i32 0, i32 1
  store ptr null, ptr %next, align 8
  
  %ptr_array = getelementptr inbounds %struct.ComplexNode, ptr %call, i32 0, i32 2, i64 0
  store ptr @global_int, ptr %ptr_array, align 8
  
  %ptr_array1 = getelementptr inbounds %struct.ComplexNode, ptr %call, i32 0, i32 2, i64 1
  store ptr @global_string, ptr %ptr_array1, align 8
  
  %inner_data = getelementptr inbounds %struct.ComplexNode, ptr %call, i32 0, i32 3, i32 0
  store ptr @global_string, ptr %inner_data, align 8
  
  %inner_size = getelementptr inbounds %struct.ComplexNode, ptr %call, i32 0, i32 3, i32 1
  store i64 20, ptr %inner_size, align 8
  
  br label %if.end

if.end:
  ret ptr %call
}

define dso_local i32 @process_complex_node(ptr noundef %node) #0 {
entry:
  %tobool = icmp ne ptr %node, null
  br i1 %tobool, label %if.then, label %if.else

if.then:
  %value = getelementptr inbounds %struct.ComplexNode, ptr %node, i32 0, i32 0
  %val = load i32, ptr %value, align 8
  
  %ptr_array = getelementptr inbounds %struct.ComplexNode, ptr %node, i32 0, i32 2, i64 0
  %int_ptr = load ptr, ptr %ptr_array, align 8
  %global_val = load i32, ptr %int_ptr, align 4
  
  %inner = getelementptr inbounds %struct.ComplexNode, ptr %node, i32 0, i32 3
  %inner_size_ptr = getelementptr inbounds %struct.Inner, ptr %inner, i32 0, i32 1
  %inner_size = load i64, ptr %inner_size_ptr, align 8
  %size_as_int = trunc i64 %inner_size to i32
  
  %result = add nsw i32 %val, %global_val
  %final_result = add nsw i32 %result, %size_as_int
  
  ; Modify the node
  %new_val = mul nsw i32 %final_result, 2
  store i32 %new_val, ptr %value, align 8
  
  ret i32 %final_result
  
if.else:
  ret i32 -1
}

define dso_local i32 @main() #0 {
entry:
  %node1 = call ptr @create_node(i32 noundef 42)
  %node2 = call ptr @create_node(i32 noundef 84)
  
  ; Link the nodes
  %next_ptr = getelementptr inbounds %struct.ComplexNode, ptr %node1, i32 0, i32 1
  store ptr %node2, ptr %next_ptr, align 8
  
  %result1 = call i32 @process_complex_node(ptr noundef %node1)
  %result2 = call i32 @process_complex_node(ptr noundef %node2)
  
  ; Access the next node through the first
  %next = load ptr, ptr %next_ptr, align 8
  %next_result = call i32 @process_complex_node(ptr noundef %next)
  
  call void @free(ptr noundef %node1)
  call void @free(ptr noundef %node2)
  
  %total = add nsw i32 %result1, %result2
  %final = add nsw i32 %total, %next_result
  ret i32 %final
}

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
