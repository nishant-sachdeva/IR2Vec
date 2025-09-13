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


#include <cxxabi.h>
#include <time.h>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>

namespace IR2Vec {

#define IR2VEC_DEBUG(X)                                                        \
  ({                                                                           \
    if (IR2Vec::debug) {                                                       \
      X;                                                                       \
    }                                                                          \
  })

using Vector = std::vector<double>;
using VocabTy = std::map<std::string, Vector>;
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
extern unsigned DIM;
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
} // namespace IR2Vec

#endif
