//===- utils.h - Helper utilities  -----------------------------*- C++ -*-===//
//
// Part of the IR2Vec Project, under the Apache License v2.0 with LLVM
// Exceptions. See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __IR2Vec_Utils__
#define __IR2Vec_Utils__

#include "llvm/ADT/SmallVector.h"
#include "llvm/Demangle/Demangle.h" //for getting function base name
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Option/Option.h>
#include "llvm/ADT/MapVector.h"

#include <stdio.h>

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/ValueTracking.h"
#include <llvm/Analysis/BasicAliasAnalysis.h>
#include <llvm/Analysis/DependenceAnalysis.h>

#include "llvm/IR/Instructions.h"
#include <llvm/Analysis/MemoryDependenceAnalysis.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instruction.h>

#include <llvm/IR/Module.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/raw_ostream.h>

#include "llvm/Passes/PassPlugin.h"
#include "llvm/Transforms/Scalar.h"


#include <cxxabi.h>
#include <time.h>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <ctime>
#include <functional>
#include <string>

using namespace llvm;
namespace IR2Vec {

#define IR2VEC_DEBUG(X)                                                        \
  ({                                                                           \
    if (IR2Vec::debug) {                                                       \
      X;                                                                       \
    }                                                                          \
  })

using Vector = std::vector<double>;
using VocabTy = std::map<std::string, Vector>;
using MapTy =
    llvm::SmallMapVector<const llvm::Instruction *,
                         llvm::SmallVector<const llvm::Instruction *, 10>, 16>;
using abi::__cxa_demangle;

extern bool fa;
extern bool sym;
extern bool printTime;
extern bool collectIR;
extern std::string iname;
extern std::string oname;
extern std::string funcName;
extern char level;
extern int cls;
extern float WO;
extern float WA;
extern float WT;
extern bool debug;
extern bool test_writeDefs;
extern bool test_reachingDefs;
extern unsigned DIM;

void _impl_collectLiveDefinitions_Walker(
    MemoryAccess *DefAccess, Instruction *targetMemLocation, AAResults &AA,
    SmallPtrSet<const Instruction *, 32> &RD, Instruction *rootInst,
    MemorySSA &MSSA);

void startCollectLiveDefinitions(MemoryAccess *StartAccess,
                                 Instruction *targetMemLocation, AAResults &AA,
                                 SmallPtrSet<const Instruction *, 32> &RD,
                                 Instruction *rootInst, MemorySSA &MSSA);

MemoryAccess* getStartAccess(Instruction *I, Instruction *memOperand,
                             MemorySSA &MSSA);

Instruction* getMemoryRoot(Instruction *memOperand);
bool accessesSameMemoryLocation(Instruction *srcInst, Instruction *targetMem,
                                AAResults &AA);

void collectLiveDefinitions(Instruction *I, Instruction *memOperand,
                            MemorySSA &MSSA, AAResults &AA,
                            SmallPtrSet<const Instruction *, 32> &RD);

void _impl_collectSSAReachingDefs(Instruction *I, MemorySSA &MSSA,
                                  AAResults &AA,
                                  SmallPtrSet<const Instruction *, 32> *RD);

void collectSSAReachingDefs(Instruction *I, MemorySSA &MSSA, AAResults &AA,
                            IR2Vec::MapTy &resultMap);

inline const llvm::Instruction* underlyingInst(const llvm::Instruction *I) {
  const llvm::Value *Ptr = llvm::getPointerOperand(I);
  if (!Ptr)
    return nullptr;

  if (const llvm::GetElementPtrInst *GEP = dyn_cast<llvm::GetElementPtrInst>(Ptr)) {
    return GEP;
  }

  // Get the deepest object in the pointer chain
  const llvm::Value *UnderlyingObj = llvm::getUnderlyingObject(Ptr);

  // If it's an instruction, that's our base
  if (const llvm::Instruction *BaseInst = dyn_cast<llvm::Instruction>(UnderlyingObj)) {
    return BaseInst;
  }

  // Otherwise, if the original pointer was an instruction, use that
  return dyn_cast<llvm::Instruction>(Ptr);
}

inline const llvm::Instruction* baseInstOf(const llvm::Instruction *I) {
  if (isa<llvm::GetElementPtrInst>(I))
    return I;

  const llvm::Instruction *Base = underlyingInst(I);

  // Keep going deeper only if we can actually go deeper
  while (Base && Base->mayReadOrWriteMemory()) {
    const Instruction *NextBase = underlyingInst(Base);
    if (!NextBase || isa<llvm::GetElementPtrInst>(NextBase)) {
      // Can't go deeper, stop here
      break;
    }
    Base = NextBase;
  }

  return Base;
}

bool rejectInstCases(llvm::Instruction *I);
std::unique_ptr<llvm::Module> getLLVMIR();
void scaleVector(Vector &vec, float factor);
// newly added
std::string getDemagledName(const llvm::Function *function);
char *getActualName(llvm::Function *function);
std::string updatedRes(IR2Vec::Vector tmp, llvm::Function *f, llvm::Module *M);
void printDependency(const llvm::Instruction* use, const llvm::Instruction* def);
void printReachingDefs(const llvm::Instruction *I, llvm::SmallVector<const llvm::Instruction*, 10> RD);
std::string getInstStr(const llvm::Instruction* I);
bool isLoadorStore(const llvm::Instruction* I);
bool isLoad(const llvm::Instruction* I);
bool isStore(const llvm::Instruction* I);

template <typename T> std::string printObject(const T *obj) {
  if(!obj) {
    std::cout << "Null Object" << std::endl;
    return std::string("Null object");
  }
  std::string output;
  llvm::raw_string_ostream rso(output);
  obj->print(rso); // Call the `print` method of the object
  rso.flush();
  return output;
}

void print_write_defs_map(
  llvm::SmallMapVector<const llvm::Instruction *,
                       llvm::SmallVector<const llvm::Instruction *, 10>, 16>
      writeDefsMapObj
);

// Generic timing wrapper function
template<typename Func>
void timeFunction(const std::string& functionName, Func&& func) {
  clock_t start = clock();
  
  // Execute the function
  func();
  
  clock_t end = clock();
  double elapsed = double(end - start) / CLOCKS_PER_SEC;
  // Replace the printf lines with:
  std::cout << "Time taken by " << functionName << " is: " << elapsed << " seconds." << std::endl;
}
} // namespace IR2Vec

#endif
